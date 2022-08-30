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

# acquire the dataset and split it into testing & training
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True) 
train_ds, test_ds = dataset['train'], dataset['test'] 

class_names = metadata.features['label'].names # store these labels for future use at the output of the model

# store how many training & testing examples there are in variables
num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples

def normalize(images, labels): 
    images = tf.cast(images, tf.float32) # change datatype
    images /= 255 # shrink to a number between 0 and 1 
    return images, labels

# normalize all of the images in the datasets so each pixel is a grayscale value between 0 and 1
train_ds.map(normalize)
test_ds.map(normalize)

# images are loaded from the disk when the dataset is first use. 
# Caching them stores them in memory so that subsequent epochs can run without loading them again
train_ds = train_ds.cache()
test_ds = test_ds.cache()

# # exploration: plot 25 images along with their labels
# plt.figure(figsize=(10, 10))
# for i, (image, label) in enumerate(test_ds.take(25)): 
#     image = image.numpy().reshape(28, 28)
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(image, cmap=plt.cm.binary)
#     plt.xlabel(class_names[label])
# plt.show()

# ============== the model =========================
# assemble layers
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)), # vectorize the input image
    tf.keras.layers.Dense(128, activation=tf.nn.relu), # one hidden layer of 128 nodes with rectified linear unit activation
    tf.keras.layers.Dense(10, # output layer has exactly enough nodes for the possible outcomes. 
                            activation=tf.nn.softmax) # softmax ensures the output emerges as a probability distribution
])

# compile the model
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(), # measures crossentropy loss between labels and predictions
    optimizer='adam', # method for reducing loss
    metrics=['accuracy']
    )

# train the model
BATCH_SIZE = 32
train_ds = train_ds.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_ds = test_ds.cache().batch(BATCH_SIZE)
# repeat() - repeat forever, except as limited by epochs in fit method
# shuffle(60000) - randomize the order as training enters a new epoch, so that the model doesn't learn anything from the order
# batch(32) - tells model.fit() to use batches of 32 images and labels when adjusting model parameters

# test the model
model.fit(train_ds, 
            epochs=5, # how many iterations to train through
            steps_per_epoch=math.ceil(num_train_examples / BATCH_SIZE) # how many batches to use in one epoch
        )

test_loss, test_accuracy = model.evaluate(test_ds, steps=math.ceil(num_test_examples / BATCH_SIZE))
print('Accuracy on test dataset:', test_accuracy)

# make predictions and explore
for test_images, test_labels in test_ds.take(1): 
    test_images, test_labels = test_images.numpy(), test_labels.numpy()
    predictions = model.predict(test_images)

print(np.argmax("predicted: ", predictions[0]))
print("actual: ", test_labels[0])


# notes: 
#   the number of epochs matters a lot when it's low. 1 epoch yields a much lower accuracy. 
# But once we get above 5, the improvement diminishes. 
#   Same thing with the number of nodes in our Dense layer. 512 is only a little bit better than 256. 
#   When we don't normalize the pixel values, there doesn't seem to be much difference... not sure why yet. 
#   A second Dense layer performs best at 64 neurons. 
# 10 neurons makes it perform very poorly, and above 128 the performance drops a little bit. 
#   Still not sure what ReLU accomplishes. 
# Adding extra layers and playing around with the neurons in each makes a marginal difference in accuracy. 