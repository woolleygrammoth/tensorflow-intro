# main libraries
import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# helper libraries
import math
import numpy as np
import matplotlib.pyplot as plt

import logging # explore logging more later
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True) # obtain dataset of interest
train_ds, test_ds = dataset['train'], dataset['test'] # split into training & testing data

class_names = metadata.features['label'].names # store these labels for future use at the output of the model

def normalize(images, labels): 
    images = tf.cast(images, tf.float32) # change datatype
    images /= 255 # shrink to a number between 0 and 1 
    return images, labels

# normalize all of the images in the datasets so each pixel is a grayscale value between 0 and 1
train_ds.map(normalize)
test_ds.map(normalize)

# images are loaded from the disk when the dataset is first use. 
# Caching them stores them in memory so that subsequent epochs can run without loading them again

# train_ds = train_ds.cache()
# test_ds = test_ds.cache()

# exploration: plot 25 images along with their labels
plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(test_ds.take(25)): 
    image = image.numpy().reshape(28, 28)
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(class_names[label])
plt.show()

