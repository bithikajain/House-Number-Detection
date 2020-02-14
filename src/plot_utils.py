import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

plt.rcParams['figure.figsize'] = (16.0, 4.0)
def plot_images(img, labels, nrows, ncols, output_dir, filename):
    """ Plot nrows x ncols images
    """
    fig, axes = plt.subplots(nrows, ncols)
    for i, ax in enumerate(axes.flat): 
        if img[i].shape == (32, 32, 3):
            ax.imshow(img[i])
        else:
            ax.imshow(img[i,:,:,0], cmap='gray')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(labels[i])
    fig.savefig(os.path.join(output_dir,filename ))

def plot_data_distribution(train_labels, test_labels, output_dir, filename):
    """
    Labels distribution in train and test data
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
    fig.suptitle('Class Distribution', fontsize=14, fontweight='bold', y=1.05)
    ax1.hist(train_labels, bins=10)
    ax1.set_title("Training set")
    ax1.set_xlim(1, 10)
    ax2.hist(test_labels, color='g', bins=10)
    ax2.set_title("Test set")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir,filename ))

def plot_train_images(images, nrows, ncols, cls_true,  output_dir, filename, cls_pred=None):
    """ Plot nrows * ncols images from images and annotate the images
    """
    # Initialize the subplotgrid
    fig, axes = plt.subplots(nrows, ncols)
    
    # Randomly select nrows * ncols images
    rs = np.random.choice(images.shape[0], nrows*ncols)
    
    # For every axes object in the grid
    for i, ax in zip(rs, axes.flat): 
        
        # Predictions are not passed
        if cls_pred is None:
            title = "True: {0}".format(np.argmax(cls_true[i]))
        
        # When predictions are passed, display labels + predictions
        else:
            title = "True: {0}, Pred: {1}".format(np.argmax(cls_true[i]), cls_pred[i])  
            
        # Display the image
        ax.imshow(images[i,:,:,0], cmap='binary')
        
        # Annotate the image
        ax.set_title(title)
        
        # Do not overlay a grid
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(os.path.join(output_dir,filename ))

def plot_confusion_metric(y_test,flat_array, output_dir,filename ):
    
    # Calculate the confusion matrix
    cm = confusion_matrix(y_true=np.argmax(y_test, axis=1), y_pred=flat_array)

    # Normalize the confusion matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100.0

    # Set the figure size
    fig = plt.figure(figsize=(12, 8))
    # Visualize the confusion matrix
    ax = sns.heatmap(cm, annot=True, cmap='Reds', fmt='.1f', square=True);
    fig.savefig(os.path.join(output_dir,filename ))

def plot_learning_rate(train_loss, valid_loss, output_dir, filename):
    
    plt.figure(figsize = (8, 6))
    plt.plot(train_loss ,'r', label = 'training loss')
    plt.plot(valid_loss, 'g' , label = 'validation loss')
    plt.xlabel('Step #')
    plt.legend()
    plt.savefig(os.path.join(output_dir,filename ))
    