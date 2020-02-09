# Street-lights-recognizer
Yolo prepared in this [project](https://github.com/Ariello05/Colab-yolov3-example)

Using OpenCV library https://opencv.org/

## Installation
* Clone
* Build for android and move into your device (usb)

## Usage
* Move your yolo config to: "/dnns/yolov3-tiny.cfg"
* Move your yolo weights to: "/dnns/yolov3-tiny.weights"

In case you use different yolo classes you might want to change code responsible for colors.
Namely Mutablelists at bottom of the [MainActivity.kt](https://github.com/Ariello05/Traffic-lights-recognizer/blob/master/app/src/main/java/com/example/streetlights/MainActivity.kt): cocoNames and scalar

## Authors
* **Paweł Dychus** - *Colors and bugfixes* - [Ariello05](https://github.com/Ariello05)
* **Patryk Łukaszuk** - *Working structure* - [Patrxon](https://github.com/patrxon)

