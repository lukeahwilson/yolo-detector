#!/usr/bin/python
# PROGRAMMER: Luke Wilson
# DATE CREATED: 2021-11-04
# REVISED DATE: 2021-??-??
# References:
#   - https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/
# NOTE:
#   - Work in progress
# PURPOSE:
#   - API to train and apply leveraged pretrained vision models for detection
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

def predict_transform(prediction, input_dim, anchors, total_classes, CUDA = True):
    '''
    Purpose:
        - Receive yolo detection layer feature map and arguments
        - Transform the feature map using arguments to represent predictions
        - Return the transformed predictions
    Arguments:
        - prediction = CNN input as feature map from previous darknet layer
            o prediction.size() = [# Images in batch, # attributes, feature map x-dimension, feature map y-dimension]
            o # Attributes is 4 box scores, 1 box objectness, 80 classes
        - input_dim = Dimensions of resized input images, set in config file
        - anchors = Anchor dimensions for yolo detection grid, set in config file
        - total_classes = Number of class outputs for classification branch of model
    Returns:
        - prediction = feature maps with transformations per math for stretching and moving anchor boxes
    '''
    batch_size = prediction.size(0)

    # NOTE: Stride is terminologically loose here. We're taking the difference in the resized input image dimension and the
    # prediction dimension at this layer to infer the approx. total stride used to reduce the original feature map size to this size.
    stride =  input_dim // prediction.size(2)

    # Save the size of the detection grid for this detection layer. Each of the 3 detection layers have a different size.
    # The first grid size is 13, for large detections, than 26, for medium, and finally 52, for small detections.
    grid_size = prediction.size(2)

    # Create number of attributes, this is the four box dimensions, the objectness, and the classes
    bbox_attrs = 5 + total_classes
    num_anchors = len(anchors)

    # Here we are flattening the prediction space for processing, turning the x by y grid into a single line of values
    prediction = prediction.view(prediction.size(0), prediction.size(1), -1)

    # We swap grid list and box rows so that attribute information for each anchor is listed in rows for each grid cell
    prediction = prediction.transpose(1,2).contiguous()

    # We further flatten, such that the last column is only the attributes, and the middle column is grid cells x anchors
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    # Sigmoid the centre x, centre y, and object confidencce per mathematical equation for anchor box transformations
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    # Create an arange from 0 to the length of one side of the constructed detection grid
    # I believe the sigmoid on the position means we can leave the grid size as a unit vector value in increments of 1
    grid = np.arange(grid_size)

    # Add the grid center offsets to the center cordinates prediction. Create a meshgrid that counts from 0 up to the grid_size
    # Copy meshgrid counting upward in rows from left to right for a, and from top to bottom in b
    a, b = np.meshgrid(grid, grid)

    # Now we flatten the first (a) mesh, creating a list of x indexes for the grid counting 0 to 9, 0 to 9... 10 times
    x_offset = torch.FloatTensor(a).view(-1,1)

    # Now we flatten the second (b) mesh, creating a list of y indexes for the grid with 10 0s, then 10 1s... 10 9s.
    y_offset = torch.FloatTensor(b).view(-1,1)

    # NOTE: This CUDA call seems out of place and also doesn't resolve CUDA errors. I'll find a better place later
    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    # So effectively, we've got two lists now that describe indexed x-y coordinates for a detection grid
    # Now we concatenate them, specifying dimension 1, so that the concatenation happens across columns
    # This results in a [grid_size x grid_size, by 2] matrix where each row is an x and y coordinate for a grid cell
    # Next we repeat the grid for the number of anchors used per box, then reshape it again with .view(-1, 2)
    # The result is a [grid_size x grid_size x 2, by 2] matrix, listing the grid coordinates of each anchor
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1,2).unsqueeze(0)

    # Wow, that is one hell of an expression above. I am absolutely certain there is a more elegant methodolgy
    # It seems to me we could loop build a matrix, and then just broadcast to it with additions/etc
    # We now sum the sigmoided box prediction coordinates with the detection grid position
    # NOTE: This matches the mathematical theory for anchor box translation sig(predx)+(anchorx)=boxx (same for y)
    prediction[:,:,:2] += x_y_offset

    # Rescaling the anchors to match the resized image. Each anchor has width and height that gets scaled by stride value
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]






    # Apply the anchors to the dimensions of the bounding box.
    anchors = torch.FloatTensor(anchors)

    # Can we not just CUDA the system at the beginning?
    if CUDA:
        anchors = anchors.cuda()

    # Once again creating another meshgrid of anchors to represent the detection grid with anchors at every cell
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)

    # NOTE: This matches the mathematical theory for anchor box transformation (anchorwid)*exp(predwid)=boxwid (same for height)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    # Sigmoid all of the class predictions to obtain probabilistic regressions of class likelihood
    prediction[:,:,5: 5 + total_classes] = torch.sigmoid((prediction[:,:,5:]))

    # Now we use the stride scalar to resize the box prediction values to that which corresponds to the size of the image
    # This way we don't have bounding boxes, sized in width and height for a small detection map, but instead for plotting on the image
    prediction[:,:,:4] *= stride

    return prediction


def write_results(prediction, confidence, total_classes, suppress_threshold = 0.4):
    '''
    Purpose:
        - Receive input containing, images, boxes, objectness scores, classes, and threshold for IOU and NMS
        - Transform the feature map using arguments to represent predictions
        - Return the transformed predictions
    Arguments:
        - prediction = combined prediction from each of the three yolo detection layers
            o 1st detection layer is 13x13 grid with 507 boxes, 2nd is 26x26 with 2028 boxes, 3rd is 52x52 with 8112 boxes
            o shape = [# images in batch, # boxes predicted per image = 10647, 85 = 4 boxes +  1 objectness + 80 classes]
        - confidence = Threshold for eliminating a box prediction based on its objectness score
        - total_classes = Number of class outputs for classification branch of model (technically not required as built into data)
        - supress_threshold = Threshold union of intersection between adjacent predictions for non-max-suppression
    Returns:
        - prediction = feature maps with transformations per math for stretching and moving anchor boxes
    '''
    # If a bounding box has an objectness score below a threshold, set the entire row (all attributes) to zero
    # Note: Need to print this section to visually confirm operation running as predicted
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2) # Shape = [#images batch, # boxes, index 4 from prediction as boolean 0 or 1]
    prediction = prediction*conf_mask # Conducts broadcasting here, where the boolean is multiplied across the matrix to zero out any objectness below confidence

    # NOTE: that index 4 is holding the objectness value. the conf_mask temporarily uses the objectness against confidence to set low confidence information to zero

    # Boxes have 4 indexes that conform to the center x coordinate, y coordinate, and box height and width
    # We convert this to top left and bottom right corner coordinates to make IOU calculation easier
    box_corner = prediction.new(prediction.shape) # I believe separating box_corner as newly instantiated matrix
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2) # Now we can make values equal to translated predictions
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2) # Repeat for y coordinate using height of box
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) # New x coordinate for bottom right with box width
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2) # Finally set bottom right with height
    prediction[:,:,:4] = box_corner[:,:,:4] # Replace 0, 1, 2, 3 box prediction values with the box corner values

    # Each image has a different number of detections and thus a different required operations.
    # Thus we cannot vectorize and compute detections in parallel and instead must loop through the predictions.
    batch_size = prediction.size(0)

    write = False

    # Now we iterate through our batch of images
    for image in range(batch_size):
        image_prediction = prediction[image]          #image Tensor

        # Pick the maximum class score which starts after objectness and the 4 box numbers and goes for number of classes
        # NOTE: I wonder if there would be improved Non Max Suppression by keeping second and third class choices around
        # to compare against surrounding detections and suppress if there are collisions between a second choice and first
        # Torch max returns tuple (max, max_indices) The one '1' takes max across row, preserving column dimension
        max_class_confidence, max_class_index = torch.max(image_prediction[:,5:5 + total_classes], 1)

        # Concatenate sequence of 10647 x 5 bounding boxes (4) + objectness (1) in first position, the 10647 x 1 classes in second position,
        # and the 10647 x 1 class confidences in the third position. We unsqueeze the values, so that we can make them a length 3 tuple
        # Then we concatenate the three individual tensors arranged in the tuple back into a single tensor, 10647 x 7, with class in column 7
        image_prediction = torch.cat((image_prediction[:,:5], max_class_confidence.unsqueeze(1), max_class_index.unsqueeze(1)), 1)

        # If no detections at all, then there would be an error, so we use try
        try:
            # Example of parsing through a bunch of crazy code below and replacing it with one simple single line.
                # non_zero_objectness =  (torch.nonzero(image_prediction[:,4]))
                # image_positive_prediction = image_prediction[non_zero_objectness,:].view(-1,7)
                # image_positive_prediction = image_prediction[torch.nonzero(image_prediction[:,4]).squeeze()].view(-1,7)

            # Search all rows along column 4 (objectness) and only return rows that were not zero'd out (set by conf_mask)
            image_positive_prediction = image_prediction[image_prediction[:,4] != 0]
            # MOVING FORWARD OUR MATRIX OF POSSIBLE DETECTIONS IS A FRACTION OF THE ORIGINAL TOTAL 10647 ATTEMPTS
        except:
            continue

        # Time to perform the Non Max Suppresion algorithm
        # First we iterate through the detected classes, one class at a time (-1 index holds the class index)
        for class_detection in unique(image_positive_prediction[:,-1]):

            # Return only the set of rows that correspond to the class that is being observed
            image_class_prediction = image_positive_prediction[image_positive_prediction[:,-1] == class_detection]

            # sort the detections to put the maximum objectness confidence at the top
            conf_sort_index = torch.sort(image_class_prediction[:,4], descending = True)[1]
            image_class_prediction = image_class_prediction[conf_sort_index]

            # Now we iterate through the number of detections made to compare intersection of union for non max suppression
            for detection in range(image_class_prediction.size(0)):
                # Get the IOUs of all boxes that come after the one we are looking at
                try:
                    ious = bbox_iou(image_class_prediction[detection].unsqueeze(0), image_class_prediction[detection+1:])
                except ValueError:
                    break
                except IndexError:
                    break

                # Zero out detections based on intersection of union compared to non max suppression threshold
                iou_mask = (ious < suppress_threshold).float().unsqueeze(1)

                # Broadcast zero'd out mask against the prediction matrix
                image_class_prediction[detection+1:] *= iou_mask

                # Search all rows along column 4 (objectness) and only return rows that were not zero'd out (set by conf_mask)
                image_class_prediction = image_class_prediction[image_class_prediction[:,4] != 0]

            # Create a batch id to associate the detections to the batch id and repeat for all detections of every class in the image
            batch_ind = image_class_prediction.new(image_class_prediction.size(0), 1).fill_(image)

            if not write:
                output = torch.cat((batch_ind, image_class_prediction),1)
                write = True
            else:
                out = torch.cat((batch_ind, image_class_prediction),1)
                output = torch.cat((output,out))

    try:
        return output
    except:
        return 0


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes


    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)

    #Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


# Function to return list of unique detected classes in the image from a list of detections
def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names


def letterbox_image(img, inp_dim):
    # I'd like to replace this with torch transform
    '''resize image with unchanged aspect ratio using padding, not sure if this is even used really'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image

    return canvas


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    img = cv2.resize(img, (inp_dim, inp_dim))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img
