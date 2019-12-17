package com.example.streetlights


import android.os.Bundle
import android.os.Environment
import android.view.SurfaceView
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import org.opencv.android.BaseLoaderCallback
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.OpenCVLoader
import org.opencv.core.*
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net
import org.opencv.imgproc.Imgproc
import org.opencv.utils.Converters
import java.util.*


class MainActivity : AppCompatActivity() , CameraBridgeViewBase.CvCameraViewListener2 {

    private lateinit var cameraBridgeViewBase: CameraBridgeViewBase
    private lateinit var baseLoaderCallback: BaseLoaderCallback

    private var startYolo = false
    private var firstTimeYolo = false
    private lateinit var tinyYolo: Net

    private var objectFound = "null"

    fun start_yolo(view: View) {

        val text = objectFound
        val duration = Toast.LENGTH_SHORT
        val toast = Toast.makeText(applicationContext, text, duration)
        toast.show()

        objectFound = "null"

        if (!startYolo) {
            startYolo = true
            if (!firstTimeYolo) {
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
        cameraBridgeViewBase.visibility = SurfaceView.VISIBLE
        cameraBridgeViewBase.setCvCameraViewListener(this)

        baseLoaderCallback = object : BaseLoaderCallback(this) {
            override fun onManagerConnected(status: Int) {
                super.onManagerConnected(status)

                when (status) {
                    SUCCESS -> cameraBridgeViewBase.enableView()
                    else -> super.onManagerConnected(status)
                }
            }
        }

    }

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame): Mat? {

        val frame = inputFrame.rgba()

        if (startYolo) {
            Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB)
            val imageBlob = Dnn.blobFromImage(
                frame, 0.00392, Size(416.0,416.0), Scalar(0.0, 0.0, 0.0),false, false
            )

            tinyYolo.setInput(imageBlob)

            val result: MutableList<Mat> = ArrayList(2)
            val outBlobNames: MutableList<String> = ArrayList()

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

            val arrayLength = confs.size
            if (arrayLength >= 1) { // Apply non-maximum suppression procedure.

                objectFound = "person"

                val nmsThresh = 0.2f
                val confidences = MatOfFloat(Converters.vector_float_to_Mat(confs))

                //val boxesArray: Array<Rect> = arrayOf(rects[0])
                val boxesArray: Array<Rect> = arrayOf(*rects.toTypedArray())

                val boxes = MatOfRect(*boxesArray)
                val indices = MatOfInt()


                Dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThresh, indices)
                // Draw result boxes:
                val ind = indices.toArray()
                for (i in ind.indices) {
                    val idx = ind[i]
                    val box: Rect = boxesArray[idx]
                    val idGuy = clsIds[idx]
                    val conf = confs[idx]

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


            }
        }
        return frame
    }

    override fun onCameraViewStarted(width: Int, height: Int) {

        if (startYolo) {
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

        cameraBridgeViewBase.disableView()
    }

    override fun onDestroy() {
        super.onDestroy()

        cameraBridgeViewBase.disableView()
    }

    private val cocoNames: MutableList<String> = arrayListOf(
        "green_light",
        "green_right",
        "green_left",
        "red_light",
        "red_right",
        "red_left",
        "yellow_light"
    )
}
