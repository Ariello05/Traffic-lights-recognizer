package com.example.streetlights

import androidx.appcompat.app.AppCompatActivity

import android.os.Environment;
import android.os.Bundle;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.*;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;

import org.opencv.dnn.Dnn;
import org.opencv.utils.Converters;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;


class MainActivity : AppCompatActivity() , CameraBridgeViewBase.CvCameraViewListener2 {

    lateinit var cameraBridgeViewBase: CameraBridgeViewBase
    lateinit var baseLoaderCallback: BaseLoaderCallback

    var startYolo = false
    var firstTimeYolo = false
    lateinit var tinyYolo: Net

    var objectFound = "null"

    fun YOLO(view: View) {

        val text = objectFound
        val duration = Toast.LENGTH_SHORT
        val toast = Toast.makeText(applicationContext, text, duration)
        toast.show()

        objectFound = "null"

        if (startYolo == false) {
            startYolo = true
            if (firstTimeYolo == false) {
                firstTimeYolo = true
                val tinyYoloCfg: String =
                    Environment.getExternalStorageDirectory().toString() + "/dnns/yolov3-tiny.cfg"
                val tinyYoloWeights: String =
                    Environment.getExternalStorageDirectory().toString() + "/dnns/yolov3-tiny.weights"
                tinyYolo = Dnn.readNetFromDarknet(tinyYoloCfg, tinyYoloWeights)
            }
        } else {
            startYolo = false
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        cameraBridgeViewBase = findViewById(R.id.CameraView)
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE)
        cameraBridgeViewBase.setCvCameraViewListener(this)

        baseLoaderCallback = object : BaseLoaderCallback(this) {
            override fun onManagerConnected(status: Int) {
                super.onManagerConnected(status)

                when (status) {
                    BaseLoaderCallback.SUCCESS -> cameraBridgeViewBase.enableView()
                    else -> super.onManagerConnected(status)
                }
            }
        }

    }

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame): Mat? {

        val frame = inputFrame.rgba()

        if (startYolo == true) {
            Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB)
            val imageBlob = Dnn.blobFromImage(
                frame, 0.00392, Size(416.0,416.0), Scalar(0.0, 0.0, 0.0),false, false
            )

            tinyYolo!!.setInput(imageBlob)

            var result: MutableList<Mat> = ArrayList(2)
            var outBlobNames: MutableList<String> = ArrayList()

            outBlobNames.add(0,"yolo_23")
            outBlobNames.add(1, "yolo_23")

            tinyYolo.forward(result, outBlobNames)

            val confThreshold = 0.3f
            val clsIds: MutableList<Int> = ArrayList()
            val confs: MutableList<Float> = ArrayList()

            val rects: MutableList<Rect> = ArrayList()

            for (i in result.indices) {
                val level = result[i]
                for (j in 0 until level.rows()) {
                    val row = level.row(j)
                    val scores = row.colRange(5, level.cols())
                    val mm = Core.minMaxLoc(scores)
                    val confidence = mm.maxVal.toFloat()
                    val classIdPoint: Point = mm.maxLoc
                    if (confidence > confThreshold) {
                        val centerX = (row[0, 0][0] * frame.cols()).toInt()
                        val centerY = (row[0, 1][0] * frame.rows()).toInt()
                        val width = (row[0, 2][0] * frame.cols()).toInt()
                        val height = (row[0, 3][0] * frame.rows()).toInt()
                        val left = centerX - width / 2
                        val top = centerY - height / 2
                        clsIds.add(classIdPoint.x.toInt())
                        confs.add(confidence)
                        rects.add(Rect(left, top, width, height))
                    }
                }
            }

            val ArrayLength = confs.size
            if (ArrayLength >= 1) { // Apply non-maximum suppression procedure.

                objectFound = "person"

                val nmsThresh = 0.2f
                val confidences = MatOfFloat(Converters.vector_float_to_Mat(confs))

                val boxesArray: Array<Rect> = arrayOf(rects[0])

                val boxes = MatOfRect(*boxesArray)
                val indices = MatOfInt()
                /*

                Dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThresh, indices)
                // Draw result boxes:
                val ind = indices.toArray()
                for (i in ind.indices) {
                    val idx = ind[i]
                    val box: Rect = boxesArray[idx]
                    val idGuy = clsIds[idx]
                    val conf = confs[idx]
                    val cocoNames: MutableList<String> = arrayListOf(
                        "a person",
                        "a bicycle",
                        "a motorbike",
                        "an airplane",
                        "a bus",
                        "a train",
                        "a truck",
                        "a boat",
                        "a traffic light",
                        "a fire hydrant",
                        "a stop sign",
                        "a parking meter",
                        "a car",
                        "a bench",
                        "a bird",
                        "a cat",
                        "a dog",
                        "a horse",
                        "a sheep",
                        "a cow",
                        "an elephant",
                        "a bear",
                        "a zebra",
                        "a giraffe",
                        "a backpack",
                        "an umbrella",
                        "a handbag",
                        "a tie",
                        "a suitcase",
                        "a frisbee",
                        "skis",
                        "a snowboard",
                        "a sports ball",
                        "a kite",
                        "a baseball bat",
                        "a baseball glove",
                        "a skateboard",
                        "a surfboard",
                        "a tennis racket",
                        "a bottle",
                        "a wine glass",
                        "a cup",
                        "a fork",
                        "a knife",
                        "a spoon",
                        "a bowl",
                        "a banana",
                        "an apple",
                        "a sandwich",
                        "an orange",
                        "broccoli",
                        "a carrot",
                        "a hot dog",
                        "a pizza",
                        "a doughnut",
                        "a cake",
                        "a chair",
                        "a sofa",
                        "a potted plant",
                        "a bed",
                        "a dining table",
                        "a toilet",
                        "a TV monitor",
                        "a laptop",
                        "a computer mouse",
                        "a remote control",
                        "a keyboard",
                        "a cell phone",
                        "a microwave",
                        "an oven",
                        "a toaster",
                        "a sink",
                        "a refrigerator",
                        "a book",
                        "a clock",
                        "a vase",
                        "a pair of scissors",
                        "a teddy bear",
                        "a hair drier",
                        "a toothbrush"
                    )

                    val intConf = (conf * 100).toInt()
                    Imgproc.putText(
                        frame,
                        cocoNames[idGuy] + " " + intConf + "%",
                        box.tl(),
                        Core.FONT_HERSHEY_SIMPLEX,
                        2.0,
                        Scalar(255.0, 255.0, 0.0),
                        2
                    )
                    Imgproc.rectangle(frame, box.tl(), box.br(), Scalar(255.0, 0.0, 0.0), 2)
                }

                 */
            }
        }
        return frame
    }

    override fun onCameraViewStarted(width: Int, height: Int) {

        if (startYolo == true) {
            val tinyYoloCfg =
                Environment.getExternalStorageDirectory().toString() + "/dnns/yolov3-tiny.cfg"
            val tinyYoloWeights =
                Environment.getExternalStorageDirectory().toString() + "/dnns/yolov3-tiny.weights"
            tinyYolo = Dnn.readNetFromDarknet(tinyYoloCfg, tinyYoloWeights)
        }

    }

    override fun onCameraViewStopped() {

    }

    override fun onResume() {
        super.onResume()

        if (!OpenCVLoader.initDebug()) {
            Toast.makeText(applicationContext, "There's a problem, yo!", Toast.LENGTH_SHORT).show()
        } else {
            baseLoaderCallback.onManagerConnected(BaseLoaderCallback.SUCCESS)
        }


    }

    override fun onPause() {
        super.onPause()

        if(cameraBridgeViewBase!=null){
            cameraBridgeViewBase.disableView()
        }
    }

    override fun onDestroy() {
        super.onDestroy()

        if (cameraBridgeViewBase!=null){
            cameraBridgeViewBase.disableView()
        }
    }
}
