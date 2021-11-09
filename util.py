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
import time, os, random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Import handmade machine learning functions for use in main
from cnn_model_functions import *
from cnn_utility_functions import *
from cnn_operational_functions  import *


def main():
    '''
    # Retrieve command line arguments to dictate model type, training parameters, and data
    # Load image datasets, process the image data, and convert these into data generators
    # Create a default naming structure to save and load information at a specified directory
    # Download a pretrained model using input arguments and attach new fully connected output Layers
    # Define criterion for loss, if training is required by the input arg, execute the following:
    #    o Prompt user for overfit training, if yes, initiate training against pretrained features
    #    o Prompt user for complete training, if yes, initiate training against pretrained features
    #    o Save the hyperparameters, training history, and training state for the overfit and full models
    # If training is no requested by the input arg, execute the following:
    #    o Load in a pretrained model's state dict and it's model_hyperparameters
    #    o Display the training history for this model
    # Provide prompt to test the model and perform and display performance if requested
    # Provide prompt to apply the model towards inference and put model to work if requested
    # Show an example prediction from the inference
    '''
    # Call ArgumentParser for user arguments and store in arg
    arg = u1_get_input_args()
    data_dir = os.path.expanduser('~') + '/Programming Data/' + arg.dir + '/'

    # Call data processor to return a dictionary of datasets, the data labels, and the class labels
    dict_datasets, dict_data_labels, dict_class_labels = u2_load_processed_data(data_dir)

    # Call data iterator to convert dictionary of datasets to dictionary of dataloaders
    dict_data_loaders = u4_data_iterator(dict_datasets)

    #Create file pathway and naming convention saving and loading files in program
    file_name_scheme =  data_dir + 'models/' + os.path.basename(os.path.dirname(data_dir))\
                    + '_' + arg.model + '_' + str(arg.layer) + 'lay'
    print(file_name_scheme)
    # Call create classifier to return a model leveraging a desired pretrained architecture, define loss criterion
    model = m1_create_classifier(arg.model, arg.layer, len(dict_datasets['train_data'].classes))
    criterion = nn.NLLLoss()

    # Define start condition hyperparameters and key running information such as elapsed training time
    # epoch_on and running_count refer to the epoch in which deeper layers started training and for how long
    model_hyperparameters = {'learnrate': arg.learn,
                         'training_loss_history': [],
                         'validate_loss_history': [],
                         'epoch_on': [],
                         'running_count': 0,
                         'weightdecay' : 0.00001,
                         'training_time' : 0}

    # If user requests load, call load checkpoint to return model and hyperparameters, then plot loaded information
    if arg.load == 'y':
        model, model_hyperparameters = m3_load_model_checkpoint(model, file_name_scheme)
        o5_plot_training_history(arg.model, model_hyperparameters, file_name_scheme)

    # If user requests train, first display an example piece of data from the processed training set
    if arg.train == 'y':
        # NOTE 1: Processed data is tensor shape [xpixel, ypixel, colour], matplotlib takes order [c, x, y], so we transpose
        # NOTE 2: Plotted images blocks function continuation, unblock requires pause to load image or image will freeze
        print('Displaying an example processed image from the training set..\n')
        plt.imshow(random.choice(dict_datasets['train_data'])[0].numpy().transpose((1, 2, 0))) # NOTE: 1
        plt.show(block=False) # NOTE: 2
        plt.pause(2)
        plt.close()

        # Call train model with model and training dataset to return trained model and hyperparameters, then plot and save
        model, model_hyperparameters = o1_train_model(model, dict_data_loaders['train_loader'],
                        dict_data_loaders['valid_loader'], arg.epoch, 0.6, model_hyperparameters, criterion)
        o5_plot_training_history(arg.model, model_hyperparameters, file_name_scheme, 'complete')

        # Prompt user to save, save the model and its hyperparameters per the naming convention
        if u5_time_limited_input('Would you like to save the model?'):
            m2_save_model_checkpoint(model, file_name_scheme, model_hyperparameters)

    # If user requests no load and no train, prompt to run an overfit training exercise and execute if requested
    # NOTE: Same as training but on an overfit dataset. overfit_model metadata references the same data as the model metadata
    if arg.train == 'n' and arg.load == 'n':
        if u5_time_limited_input('Check model can overfit small dataset?'):
            overfit_model, overfit_model_hyperparameters = o1_train_model(model, dict_data_loaders['overfit_loader'],
                            dict_data_loaders['valid_loader'], arg.epoch, 0.9, model_hyperparameters, criterion)
            o5_plot_training_history(arg.model, overfit_model_hyperparameters, file_name_scheme, 'overfit')

    # If user requests load, or has requested training and training has completed, the model is ready for predictions
    if arg.train == 'y' or arg.load == 'y':
        print('The model is ready to provide predictions\n')

        # Prompt to test the model's performance
        # Gives the testing data loader to the validation function and returns performance
        if u5_time_limited_input('Would you like to test the model?'):
            t0 = time.time()
            test_count_correct, ave_test_loss = o3_model_no_backprop(model, dict_data_loaders['test_loader'], criterion)
            print('\nTesting Loss: {:.3f}.. '.format(ave_test_loss),
                'Testing Accuracy: {:.3f}'.format(test_count_correct / len(dict_data_loaders['test_loader'].dataset)),
                'Runtime - {:.0f} seconds\n'.format((time.time() - t0)))

        # Prompt the user to use the model for inference
        # Gives an unlabeled dataloader to a predict function and returns predictions
        if u5_time_limited_input('Would you like to use the model for inference?'):
            t1 = time.time()
            dict_prediction_results = o6_predict_data(model, dict_data_loaders['predict_loader'],
                            dict_data_labels, dict_class_labels)
            print('Runtime - {:.0f} seconds\n'.format((time.time() - t1)),
                            [dict_prediction_results[key][0][0] for key in dict_prediction_results])
            o7_show_prediction(data_dir, dict_prediction_results)


if __name__ == "__main__":
    main()
