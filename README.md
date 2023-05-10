# CPPND: Capstone Image Classification Runtime System (ICARUS) Repo

This is the repo for my Capstone project in the [Udacity C++ Nanodegree Program](https://www.udacity.com/course/c-plus-plus-nanodegree--nd213).

For this project I decided to implement a C++ application for performing image classification inference on a set of images, utilizing the ONNX Runtime and OpenCV under the hood.

The application continuously reads ("captures") the images from the respective folder (assets/images) and performs inference on each of them. The image and the inference result (execution time and the respective label from assets/label) is displayed and updated on a pop-up window.

The images are a subset of the "ImageNet Large Scale Visual Recognition Challenge 2012" test dataset (ILSVRC2012_test).

The model used for the Image Classification is MobileNetV2 (located in assets/model).

The aim of the project is to support easy extension to other image data providers and models via
abstraction layers (e.g. image_preproprocessor and model handler classes).

Software was tested on an Ubuntu 22.04 x86_64 machine.

For Linux all dependencies are automatically installed by running the setup.sh script
For other OS and platform architectures you will need to check the setup.sh script and replicate
the steps on your corresponding OS/platform consulting the thirdparty documentation.

## Dependencies for Running Locally
* cmake >= 3.7
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* opencv-4.x
  * Linux: Automatically installed by running setup.sh
  * Other OS/platforms: Please follow steps in setup.sh and consult the appropriate opencv documentation for your OS and platform
* onnxruntime 1.13.1
  * Linux: Automatically installed for a x86_64 CPU architecture by running setup.sh
  * Other OS/platforms: Please follow steps in setup.sh and consult the appropriate onnx runtime documentation for your OS and platform

## Basic Build Instructions

1. Clone this repo.
2. Install all required third party dependencies: `./setup.sh`
3. Compile: `./build_env.sh`
4. Run: `./Icarus`