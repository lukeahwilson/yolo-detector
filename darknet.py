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
import time, os, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from util import *

def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    # This image resize function is not connected to the darknet grid sizing, will error if doesn't match net height/width
    img = cv2.resize(img, (416, 416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_


def parse_cfg(cfgfile):
    """
    Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list

    """
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')                        # store the lines in a list
    lines = [x for x in lines if len(x) > 0]               # get read of the empty lines
    lines = [x for x in lines if x[0] != '#']              # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]           # get rid of fringe whitespaces

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":               # This marks the start of a new block
            if len(block) != 0:          # If block is not empty, implies it is storing values of previous block.
                blocks.append(block)     # add it the blocks list
                block = {}               # re-init the block
            block["type"] = line[1:-1].rstrip()
        else:
            key,value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks

def create_modules(blocks):
    net_info = blocks[0]     #Captures the information about the input and pre-processing
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        #check the type of block
        #create a new module for the block
        #append to module_list
        if (x["type"] == "convolutional"):
            #Get the info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters= int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            #Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)

            #Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            #Check the activation.
            #It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), activn)

        #If it's an upsampling layer
        #We use Bilinear2dUpsampling
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode = "bilinear")
            module.add_module("upsample_{}".format(index), upsample)

        #If it is a route layer
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            #Start  of a route
            start = int(x["layers"][0])
            #end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            #Positive anotation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters= output_filters[index + start]

        #shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

                #Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]


            anchors = x["anchors"].split(",") # Grab the string of anchors from the yolo layer and split them by the commas
            anchors = [int(a) for a in anchors] # Grab only the integers out of the strings of anchor number values
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)] # Reset list in groupings of 2
            anchors = [anchors[i] for i in mask] # I guess we only want to use the first three anchors for detection

            detection = DetectionLayer(anchors) # Calls detection layer class, creating an object with attached anchors
            module.add_module("Detection_{}".format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}   # We cache the outputs for the route layer

        write = 0     # Set flag to zero to wait for a detection before initializing collector
        for i, module in enumerate(modules):
            module_type = (module["type"])

            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)

            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                # This part appears to simply convert the reference point to be relative to the current layer
                # This is because if the layer value is positive, than we concatenate the absolute index of the layer
                # But if the layer value is negative, it is a relative index, and we move to that layer. This makes it relative either way
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                # layers length is 1, make x equal to the cached output from the called route layer
                if len(layers) == 1:
                    x = outputs[i + (layers[0])] # (i is current layer, + the relative index)

                else: # I really don't like the structure of this statement, it depends a lot on config structure and isn't generalized)
                    if (layers[1]) > 0: # The second layer goes through the same absolute to relative conversion
                        layers[1] = layers[1] - i

                    # Note that concatenate is not the same as add, the two layers remain separate with separate weight matrices
                    # combine the multiple route layers if the route length is greater than 1 and concatenate it to make x
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1)

            elif  module_type == "shortcut": # This is so similar to the route layer except for summing the x with the previous layers x
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]

            elif module_type == 'yolo':


                anchors = self.module_list[i][0].anchors
                # Get the input dimensions
                inp_dim = int (self.net_info["height"])

                # Get the number of classes
                num_classes = int (module["classes"])

                #Transform
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)

                # initialize the collector cycle for concatenating detections
                if not write:              # if no collector has been intialised.
                    detections = x
                    write = 1

                # if already initialized, concatenate to collect detections
                else:
                    detections = torch.cat((detections, x), 1)

            # Cache the output for use in shortcut and route layers
            outputs[i] = x

        return detections

    def load_weights(self, weightfile):

        #Open the weights file
        fp = open(weightfile, "rb")

        #The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype = np.float32)


        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            #If module_type is convolutional load weights
            #Otherwise ignore.
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                # NEED TO REREAD THIS AND GET THIS CODE DIALED (NOTE TO SELF)
                if (batch_normalize):
                    # Biases include the bn bias, bn weight, bn running mean, bn running var, conv weight

                    bn = model[1]

                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    #Load the weights
                    #Here we load a round of weights starting at index 0 and going to 0 + number
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # We add the number that we pulled to allow us to move to next round of biases
                    #Then we repeat this process for each portion of the weights comprising this layer
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    #Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    # Biases include the conv bias, conv weight
                    #Number of biases
                    num_biases = conv.bias.numel()

                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)

                # Either way, we then also load conv weights
                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()

                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

blocks = parse_cfg("cfg/yolov3.cfg")
# print(create_modules(blocks))


model = Darknet("cfg/yolov3.cfg")
model.cuda()
inp = get_test_input()
inp = inp.cuda()
pred = model(inp, torch.cuda.is_available())
print (pred)
print(pred.shape)
model.load_weights("yolov3.weights")
