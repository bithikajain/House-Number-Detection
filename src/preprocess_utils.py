#!/usr/bin/env python

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from skimage import color
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import h5py
import tensorflow as tf
#import torch

plt.rcParams['figure.figsize'] = (16.0, 4.0)
###############################################################################
###############################################################################

#device = torch.device("cpu")

###############################################################################
###############################################################################



def load_data(path):
    """ Helper function for loading a MAT-File"""
    data = loadmat(path)
    return data['X'], data['y']

def convert_labels_10to0(labels):
    labels[labels ==10] = 0


def rgb2gray(images):
    """converting in human readable format 
       grayscale conversion to reduce the amount of data to be processed 
       Y =  0.2990R + 0.5870G + 0.1140B"""
    return np.expand_dims(np.dot(images, [0.2990, 0.5870, 0.1140]), axis=3)

def normalize(train_greyscale, test_greyscale, val_greyscale):
    """Normalization refers to normalizing the data dimensions so that they
       are of approximately the same scale. Divide each dimension by its 
       standard deviation, once it has been zero-centered.

       Normalization done using mean and std on training data
    """

    # Calculate the mean on the training data
    train_mean = np.mean(train_greyscale, axis=0)

    # Calculate the std on the training data
    train_std = np.std(train_greyscale, axis=0)

    # Subtract it equally from all splits
    train_greyscale_norm = (train_greyscale - train_mean) / train_std
    test_greyscale_norm = (test_greyscale - train_mean)  / train_std
    val_greyscale_norm = (val_greyscale - train_mean) / train_std
    return train_greyscale_norm, test_greyscale_norm, val_greyscale_norm
    
def one_hot_labels(y_train, y_test, y_val ):
    enc = OneHotEncoder().fit(y_train.reshape(-1,1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()
    y_val = enc.transform(y_val.reshape(-1, 1)).toarray()
    return y_train, y_test, y_val

def store_data(filename, 
                output_dir,
                train_greyscale_norm, test_greyscale_norm, val_greyscale_norm,
                y_train, y_test, y_val  ):
    h5f = h5py.File(os.path.join(output_dir, filename), 'w')

    # Store the datasets
    h5f.create_dataset('X_train', data=train_greyscale_norm)
    h5f.create_dataset('y_train', data=y_train)
    h5f.create_dataset('X_test', data=test_greyscale_norm)
    h5f.create_dataset('y_test', data=y_test)
    h5f.create_dataset('X_val', data=val_greyscale_norm)
    h5f.create_dataset('y_val', data=y_val)

    # Close the file
    h5f.close()

def load_preprocessed_data(filename, input_dir):
    # Open the file as readonly
    h5f = h5py.File(os.path.join(input_dir, filename), 'r')

    # Load the training, test and validation set
    X_train = h5f['X_train'][:]
    y_train = h5f['y_train'][:]
    X_test = h5f['X_test'][:]
    y_test = h5f['y_test'][:]
    X_val = h5f['X_val'][:]
    y_val = h5f['y_val'][:]

    # Close this file
    h5f.close()
    return X_train, y_train, X_test, y_test, X_val, y_val


def prepare_log_dir(TENSORBOARD_SUMMARIES_DIR):
    '''Clears the log files then creates new directories to place
        the tensorbard log file.''' 
    if tf.gfile.Exists(TENSORBOARD_SUMMARIES_DIR):
        tf.gfile.DeleteRecursively(TENSORBOARD_SUMMARIES_DIR)
    tf.gfile.MakeDirs(TENSORBOARD_SUMMARIES_DIR)
    
def get_batch(X, y, batch_size=512):
    """
    Dynamically get batches
    """
    for i in np.arange(0, y.shape[0], batch_size):
        end = min(X.shape[0], i + batch_size)
        yield(X[i:end],y[i:end])


