# Yolo-Powered-Detector
A object detecting neural network powered by the yolo architecture and leveraging the PyTorch framework and associated libraries. The yolo architecture is being built and formatted directly from the detailed and useful blog by Ayoosh Kathuria and his associated github repository. Once mastering this tutorial on object detection I will be adding commits to personalize the model and integrate it into my own application.

## PURPOSE:
API to train and apply the yolo-architecture for object detection and classification

## REQUIREMENTS:
- Yolo architecture is downloaded and can be trained on a dataset by user
- The number architecture and number of outputs is customizable by the user
- The deeper convolutional layers are unfrozen for a period of time during training for tuning
- User can load a model and continue training or move directly to inference
- Saved trained model information is stored in a specific folder with a useful naming convention
- There are time-limited prompts that allow the user to direct processes as needed
- Training performance can be tested before moving onward to inference if desired
- Predictions are made using paralleled batches and are saved in a results dictionary

## HOW TO USE:
- Pending Project Completion

## Table Of Contents

### darknet
-

### cfg
-

## Credits
This repository is being built off of Ayoosh Kathuria's shared github repository for implementing the yolo architecture https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch. This repository is inspired by ideas that I learned from Stanford's online lecture series CS231n Convolutional Neural Networks for Visual Recognition.

## Dependencies
Please see the `requirements.txt` file or the environment `env-detector-pytorch.yaml` file for minimal dependencies required to run the repository code.

## Install
To install these dependencies with pip, you can issue `pip3 install -r requirements.txt`
To install these dependencies with conda, use `conda env create --file env-detector-pytorch.yaml`
