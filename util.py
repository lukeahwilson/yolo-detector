#!/usr/bin/python
# PROGRAMMER: Luke Wilson
# DATE CREATED: 2021-11-04
# REVISED DATE: 2021-??-??
# References:
#   - https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/
# PURPOSE:
#   - API to train and apply leveraged pretrained vision models for classification
# REQUIREMENTS:
#   - Pretained model is downloaded and can be trained on a dataset by user
#   - The number of attached fully connected layers is customizable by the user
#   - The deeper convolutional layers are unfrozen for a period of time during training for tuning
#   - User can load a model and continue training or move directly to inference
#   - Saved trained model information is stored in a specific folder with a useful naming convention
#   - There are time-limited prompts that allow the user to direct processes as needed
#   - Training performance can be tested before moving onward to inference if desired
#   - Predictions are made using paralleled batches and are saved in a results dictionary
# HOW TO USE:
#   - If no model has been trained and saved, start by training a model
#   - Store data in folders at this location: os.path.expanduser('~') + '/Programming Data/'
#   - For training, 'train' and 'valid' folders with data are required in the data_dir
#   - For overfit testing, an 'overfit' folder with data is required in the data_dir
#   - For performance testing, a 'test' folder with data is required in the data_dir
#   - For inference, put data of interest in a 'predict' folder in the data_dir
#   - For saving and loading models, create a 'models' folder in the data_dir
##

# Import libraries
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
    print('prediction 0', prediction[0])
    batch_size = prediction.size(0)

    print('prediction 2', prediction[2])

    # determine what stride was used in total to move the original image to the new prediction size
    stride =  inp_dim // prediction.size(2)

    # construct the detection grid to enable multiple detections, stride is
    grid_size = inp_dim // stride

    # create number of attributes, this is the four box dimensions, the objectness, and the classes
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    # here we are flattening the prediction space for processing, leaving the depth as rows
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    # Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    # Add the grid center offsets to the center cordinates prediction.
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    # Apply the anchors to the dimensions of the bounding box.
    # log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

    # Now we use the stride scalar to resize the prediction values to that which corresponds to the size of the image
    # This way we don't have bounding boxes, sized in width and height for a small detection map, but instead for plotting on the image
    prediction[:,:,:4] *= stride

    return prediction
