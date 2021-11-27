# Yolo-Powered-Detector
A object detecting neural network powered by the yolo architecture and leveraging the PyTorch framework and associated libraries.

## PURPOSE:
An exercise to build a yolov3 detector class from scratch using the open source config file
The yolo architecture is built by learning from and following along the blog by Ayoosh Kathuria and his associated github repository.
Once mastering this tutorial I will be aiming to put this knowledge towards applying detection and deep learning towards whatever tasks interest me next.

## REFLECTION:
- I worked through the notes meticulously to learn and understand the way the architecture gets built and detection is conducted.
- I implemented unique solutions in areas to simplify code. I heavily annotated code to solidify my understanding of the program.
- The journey of implementing a yolo detector from scratch taught me how to turn the concept of detection into a functioning model.
- I have obtained a strong foundational understanding of the yolo architecture, and by extension, detection architectures in general.

## Table Of Contents
### darknet_model_functions
- The darknet is the nomenclature for the yolo class for detection. This file contains necessary code to parse the yolo config into the functioning classes.

### darknet_operational_functions
- Operational functions conducting key detection processes such as intersection of union, non max suppression, and data manipulation for detection for the darknet.

### darknet_detector
- A detector script that runs the model functions to load the yolo architecture, then runs the operational functions and prints and saves detections from inputs.

### cfg
- Config file listing parameters for layers that comprise the yolov3 architecture. This file is to be parsed into an object-oriented class with yolo methods.

## Credits
This repository is being built off of Ayoosh Kathuria's shared github repository for implementing the yolo architecture https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch.
This repository is inspired by ideas that I learned from Stanford's online lecture series CS231n Convolutional Neural Networks for Visual Recognition.

## Dependencies
Please see the `requirements.txt` file or the environment `env-detector-pytorch.yaml` file for minimal dependencies required to run the repository code.

## Install
To install these dependencies with pip, you can issue `pip3 install -r requirements.txt`
To install these dependencies with conda, use `conda env create --file env-detector-pytorch.yaml`
