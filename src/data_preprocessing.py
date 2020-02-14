#!/usr/bin/env python

import os
import sys
sys.path.append('/home/bithika/src/House-Number-Detection')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from skimage import color
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
#import torch
import argparse
import h5py

plt.rcParams['figure.figsize'] = (16.0, 4.0)
###############################################################################
###############################################################################

#device = torch.device("cpu")


from preprocess_utils import *
from plot_utils import *


###############################################################################
###############################################################################
#               Argument Parsing
#
parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--base-dir', type=str,
                    default='/home/bithika/src/House-Number-Detection', help='Input base directory ')
parser.add_argument('--train-dir', type=str,
                    default='/home/bithika/src/House-Number-Detection/data/raw/train_32x32.mat', help='Input data directory')
parser.add_argument('--test-dir', type=str,
                    default='/home/bithika/src/House-Number-Detection/data/raw/test_32x32.mat', help='Input data directory')
parser.add_argument('--output-dir', type=str,
                    default='/home/bithika/src/House-Number-Detection/reports', help='Input data directory')
parser.add_argument('--processed-data-dir', type=str,
                    default='/home/bithika/src/House-Number-Detection/data/processed', help='processed data directory')
parser.add_argument('--validation-data-fraction', type=float,
                    default=0.1, help='validation dataset split fraction (default: 0.1)')
args = parser.parse_args()
###############################################################################
###############################################################################
#              Load dataset
#
# Reading the .mat files 
X_train, y_train = load_data(args.train_dir)
X_test, y_test = load_data(args.test_dir)

print("Training Set", X_train.shape, y_train.shape)
print("Test Set", X_test.shape, y_test.shape)

# Calculate the total number of images
num_images = X_train.shape[0] + X_test.shape[0]

print("Total Number of Images", num_images)
# Transpose image arrays
# (width, height, channels, size) -> (size, width, height, channels)
X_train, y_train = X_train.transpose((3,0,1,2)), y_train[:,0]
X_test, y_test = X_test.transpose((3,0,1,2)), y_test[:,0]
print("Training Set", X_train.shape)
print("Test Set", X_test.shape)
print('')

# Plot some training set images
plot_images(X_train, y_train, 2, 8, args.output_dir, 'train_images.png')
# Plot some test set images
plot_images(X_test, y_test, 2, 8, args.output_dir, 'test_images.png')
# check for unique labesl
print(np.unique(y_train))
# data distribution
plot_data_distribution(y_train, y_test, args.output_dir, 'class_distribution.png')
# distributions are skewed in the positive direction i.e lesser data on the higher values

convert_labels_10to0(y_train)
convert_labels_10to0(y_test)
# check for unique labesl
print(np.unique(y_train))

# split training data into train and validation 
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.13, random_state=7, stratify = y_train)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=args.validation_data_fraction, random_state=7)

plot_data_distribution(y_train, y_val, args.output_dir, 'train_val_class_distribution.png')
print(y_train.shape, y_val.shape, y_test.shape)
# convert to float for numpy computation 

train_greyscale = rgb2gray(X_train).astype(np.float32)
test_greyscale = rgb2gray(X_test).astype(np.float32)
val_greyscale = rgb2gray(X_val).astype(np.float32)
print("Training Set", train_greyscale.shape)
print("Validation Set", val_greyscale.shape)
print("Test Set", test_greyscale.shape)
print('')
# remove RGB train, test and val set from RAM 
del X_train, X_val, X_test

plot_images(train_greyscale, y_train, 1, 10,args.output_dir, 'train_images_greyscale.png' )
# Normalisation 
# Liang et al. 2015 report that the pre-processed the images by removing the per-pixel mean value calculated over 
#the entire set.
#Goodfellow et al. 2013 report that they subtract the mean from every image.
train_greyscale_norm, test_greyscale_norm, val_greyscale_norm = normalize(train_greyscale, test_greyscale, val_greyscale)
plot_images(train_greyscale, y_train, 1, 10, args.output_dir, 'train_images_greyscale_norm.png' )
#one hot label encoding
y_train, y_test, y_val = one_hot_labels(y_train, y_test, y_val )
print("Training set", y_train.shape)
print("Validation set", y_val.shape)
print("Test set", y_test.shape)

store_data('SVHN_grey.h5', 
                args.processed_data_dir,
                train_greyscale_norm, test_greyscale_norm, val_greyscale_norm,
                y_train, y_test, y_val)

                