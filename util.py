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

    batch_size = prediction.size(0)

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


# Function to take prediction and objectness scores, number of classes, and threshold for IOU and NMS
# The prediction shape is [# images in batch, # boxes predicted per image = 10647, 85 = 1 objectness, 4 boxes, 80 classes]
def write_results(prediction, confidence, num_classes, nms_conf = 0.4):

    # If a bounding box has an objectness score below a threshold, set the entire row (all attributes) to zero
    # Note: Need to print this section to visually confirm operation running as predicted
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask

    # Boxes have 4 indexes that conform to the center x coordinate, y coordinate, and box height and width
    # We convert this to top left and bottom right corner coordinates to make IOU calculation easier
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2)
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]

    # Each image has a different number of detections and thus a different required operations.
    # Thus we cannot vectorize and compute detections in parallel and instead must loop through the predictions.
    batch_size = prediction.size(0)

    write = False

    for ind in range(batch_size):
        image_pred = prediction[ind]          #image Tensor
        #confidence threshholding
        #NMS

        # Pick the maximum class score which starts after objectness and the 4 box numbers and goes for number of classes
        # NOTE: I wonder if there would be improved Non Max Suppression by keeping second and third class choices around
        # to compare against surrounding detections and suppress if there are collisions between a second choice and first
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)

        # Get rid of bounding box scores with object confidence lower than threshhold
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))
        # If no detections at all, then there would be an error, so we use try
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue

        #For PyTorch 0.4 compatibility
        #Since the above code with not raise exception for no detection
        #as scalars are supported in PyTorch 0.4
        if image_pred_.shape[0] == 0:
            continue

        #Get the various classes detected in the image
        img_classes = unique(image_pred_[:,-1]) # -1 index holds the class index

        #Time to perform the Non Max Suppresion algorithm
        for cls in img_classes:
            # NOTE: I need to read this entire thing in details and really understand it better

            #get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)

            #sort the detections such that the entry with the maximum objectness
            #confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)   #Number of detections

            for i in range(idx):
                #Get the IOUs of all boxes that come after the one we are looking at
                #in the loop
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break

                except IndexError:
                    break

                #Zero out all the detections that have IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask

                #Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)

# Function to return list of unique detected classes in the image from a list of detections
def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res
