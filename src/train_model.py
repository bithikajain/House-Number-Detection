
from __future__ import absolute_import
from __future__ import print_function
import os
import sys
sys.path.append('/home/bithika/src/House-Number-Detection')
import time
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from skimage import color
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import argparse
import h5py

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
plt.rcParams['figure.figsize'] = (16.0, 4.0)

from preprocess_utils import load_preprocessed_data, prepare_log_dir, get_batch

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
parser.add_argument('--summary-dir', type=str,
                    default='/home/bithika/src/House-Number-Detection/models', help='summary directory ')

args = parser.parse_args()
###############################################################################
###############################################################################


max_epochs = 2
batch_size = 512

#Discarding or fuse % of neurons in Train mode
discard_per = 0.7


TENSORBOARD_SUMMARIES_DIR = os.path.join(args.summary_dir, 'svhn_classifier_logs')

print('Loading data...')
X_train, y_train, X_test, y_test, X_val, y_val = load_preprocessed_data('SVHN_grey.h5', 
                                                                        args.processed_data_dir)
num_examples = X_train.shape[0]
print('Training set', X_train.shape, y_train.shape)
print('Validation set', X_val.shape, y_val.shape)
print('Test set', X_test.shape, y_test.shape)                                           

### placeholder variable
comp = 32*32
tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here
x = tf.placeholder(tf.float32, shape = [None, 32, 32, 1], name='Input_Data')
y = tf.placeholder(tf.float32, shape = [None, 10], name='Input_Labels')
y_cls = tf.argmax(y, 1)

discard_rate = tf.placeholder(tf.float32, name='Discard_rate')
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

prepare_log_dir(TENSORBOARD_SUMMARIES_DIR)

###############################################################################

def cnn_model_fn(features):
    """Model function for CNN.
       INPUT -> [CONV -> RELU -> CONV -> RELU -> POOL] -> DROPOUT -> [FC -> RELU] -> FC
    """
    
      # Input Layer
    input_layer = tf.reshape(features, [-1, 32, 32, 1], name='Reshaped_Input')

      # Convolutional Layer #1
    #with tf.name_scope('Conv1 Layer + ReLU'):
    
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

      # Pooling Layer #1
    #with tf.name_scope('Pool1 Layer'):
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

      # Convolutional Layer #2 and Pooling Layer #2
    #with tf.name_scope('Conv2 Layer + ReLU'): 
    conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=64,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)
        
    #with tf.name_scope('Pool2 Layer'):
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

      # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=256, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
         inputs=dense, rate=discard_rate)

      # Logits Layer
    #with tf.name_scope('Logits Layer'):
    logits = tf.layers.dense(inputs=dropout, units=10)

    return logits
###############################################################################
###############################################################################
#
#   Prediction and Optimizer
#
#
#with tf.name_scope('Model Prediction'):
prediction = cnn_model_fn(x)
prediction_cls = tf.argmax(prediction, 1)
#with tf.name_scope('loss'):
loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(
   onehot_labels=y, logits=prediction))
    #tf.summary.scalar('loss', loss)
    
#with tf.name_scope('Adam Optimizer'):
optimizer = tf.train.AdamOptimizer().minimize(loss)

###############################################################################
###############################################################################
#
#   Accuracy
#
#
# Predicted class equals the true class of each image?
correct_prediction = tf.equal(prediction_cls, y_cls)

# Cast predictions to float and calculate the mean
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

###############################################################################
###############################################################################
#
#   Tensorflow session
#
#
sess = tf.Session()
sess.run(tf.global_variables_initializer())

### save model ckpts 
saver = tf.train.Saver()
save_dir = os.path.join(args.summary_dir, 'checkpoints/')

# Create directory if it does not exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
save_path = os.path.join(save_dir, 'svhn_single_greyscale')

###############################################################################
###############################################################################
#
#        restore variable
# 
#
#saver.restore(sess=session, save_path=save_path)

#with tf.Session() as sess:
 #   sess.run(tf.global_variables_initializer())
    
## To calculate total time of training
train_loss = []
valid_loss = []
start_time = time.time()
for epoch in range(max_epochs):
    print ('Training .........')
    epoch_loss = 0
    print ()
    print ('Epoch ', epoch+1 , ': ........ \n')
    step = 0   
    
    ## Training epochs ....
    for (epoch_x , epoch_y) in get_batch(X_train, y_train, batch_size):
        _, train_accu, c = sess.run([optimizer, accuracy, loss], feed_dict={x: epoch_x, y: epoch_y, discard_rate: discard_per})
        train_loss.append(c)
    
        if(step%40 == 0):
            print ("Step:", step, ".....", "\nMini-Batch Loss   : ", c)
            print('Mini-Batch Accuracy :' , train_accu*100.0, '%')

            ## Validating prediction and summaries
            accu = 0.0
            for (epoch_x , epoch_y) in get_batch(X_val, y_val, 512):                            
                correct, _c = sess.run([correct_prediction, loss], feed_dict={x: epoch_x, y: epoch_y, discard_rate: 0.0})
                valid_loss.append(_c)
                accu+= np.sum(correct[correct == True])
            print('Validation Accuracy :' , accu*100.0/y_val.shape[0], '%')
            print ()
        step = step + 1


    print ('Epoch', epoch+1, 'completed out of ', max_epochs)

    
## Calculate net time
time_diff = time.time() - start_time

## Testing prediction and summaries
accu = 0.0
for (epoch_x , epoch_y) in get_batch(X_test, y_test, 512):
    correct = sess.run([correct_prediction], feed_dict={x: epoch_x, y: epoch_y, discard_rate: 0.0})
    accu+= np.sum(correct[correct == True])
print('Test Accuracy :' , accu*100.0/y_test.shape[0], '%')
print("Time usage: " + str(timedelta(seconds=int(round(time_diff)))))
print ()

saver.save(sess=sess, save_path=save_path)