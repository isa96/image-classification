# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
import timeit

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print(
      '\n\nThis error most likely means that this notebook is not '
      'configured to use a GPU.  Change this in Notebook Settings via the '
      'command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n')
  raise SystemError('GPU device not found')

def cpu():
  with tf.device('/cpu:0'):
    random_image_cpu = tf.random.normal((100, 100, 100, 3))
    net_cpu = tf.keras.layers.Conv2D(32, 7)(random_image_cpu)
    return tf.math.reduce_sum(net_cpu)

def gpu():
  with tf.device('/device:GPU:0'):
    random_image_gpu = tf.random.normal((100, 100, 100, 3))
    net_gpu = tf.keras.layers.Conv2D(32, 7)(random_image_gpu)
    return tf.math.reduce_sum(net_gpu)
  
# We run each op once to warm up; see: https://stackoverflow.com/a/45067900
cpu()
gpu()

# Run the op several times.
print('Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images '
      '(batch x height x width x channel). Sum of ten runs.')
print('CPU (s):')
cpu_time = timeit.timeit('cpu()', number=10, setup="from __main__ import cpu")
print(cpu_time)
print('GPU (s):')
gpu_time = timeit.timeit('gpu()', number=10, setup="from __main__ import gpu")
print(gpu_time)
print('GPU speedup over CPU: {}x'.format(int(cpu_time/gpu_time)))

"""# Importing libraries"""

import pandas as pd
import numpy as np
import os
import seaborn as sns
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Activation
from sklearn.model_selection import train_test_split

from google.colab import files

"""# Kaggle setup"""

# Install kaggle
!pip install -q kaggle

# Upload json file downloaded from kaggle account
uploaded = files.upload()

# Create kaggle directory and move the uploaded file to new folder
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle

# overwrite read permissions to be accessible on Google Colab
!chmod 600 /root/.kaggle/kaggle.json

# Download datasets
!kaggle datasets download -d zalando-research/fashionmnist

# Extract the zip file
!unzip -q /content/fashionmnist.zip -d .

"""# Retrieve datasets and manage files"""

data_train = pd.read_csv('/content/fashion-mnist_train.csv')
data_train.head(6)

data_test = pd.read_csv('/content/fashion-mnist_test.csv')
data_test.head(6)

# Viewing data in the dataset
print('Dataset:')

print(f'Total images in the train dataset: {data_train.shape[0]}')
print(f'Total images in the test dataset: {data_test.shape[0]}')

data_training = data_train.drop('label',axis=1)
value_training = data_train['label']

data_testing = data_test.drop('label',axis=1)
value_testing = data_test['label']

data_training = data_training.to_numpy()
value_training = value_training.to_numpy()

data_testing = data_testing.to_numpy()
value_testing = value_testing.to_numpy()

data_training = data_training.reshape(-1,28,28)
data_testing = data_testing.reshape(-1,28,28)

"""# Dataset information

**Content**
> Each image in this dataset is 28 pixels in height and 28 pixels in width, for a total of 784 pixels.



**Labels**
> Each training and test example is assigned to one of the following labels:
*   0: T-shirt/top
*   1: Trouser
*   2: Pullover
*   3: Dress
*   4: Coat
*   5: Sandal
*   6: Shirt
*   7: Sneaker
*   8: Bag
*   9: Ankle boot


"""

labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal'
              ,'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

"""# plot a few samples from each dataset"""

plt.figure(figsize=(12,12))

for i in range(0, 25):
    plt.subplot(5,5,i+1)
    plt.imshow(data_training[i])
    plt.title(labels[value_training[i]])
    plt.axis("off")

plt.tight_layout()

plt.show()

labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal' ,'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(12,12))

for i in range(0, 25):
    plt.subplot(5,5,i+1)
    plt.imshow(data_testing[i])
    plt.title(labels[value_testing[i]])
    plt.axis("off")

plt.tight_layout()

plt.show()

data_training = data_training/255
data_testing = data_testing/255

data_training = data_training.reshape(-1,28,28,1)
data_testing = data_testing.reshape(-1,28,28,1)

"""# Splitting data"""

x_train, x_val, y_train, y_val = train_test_split(data_training, value_training, test_size = 0.2, stratify=value_training, random_state=2)

"""# Building model"""

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,kernel_size=3, activation='relu', input_shape=(28, 28, 1),padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2),padding='same'),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64,kernel_size=3, activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2),padding='same'),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, kernel_size=3,activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2),padding='same'),
    tf.keras.layers.Dropout(0.25),


    tf.keras.layers.Conv2D(128, kernel_size=3,activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2),padding='same'),
    tf.keras.layers.Dropout(0.25),


    tf.keras.layers.Conv2D(128, kernel_size=3,activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2),padding='same'),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()

# Use sparse_categorical_crossentropy because it is a multiclass category
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

"""# Creating callbacks function"""

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.93 and logs.get('val_accuracy')>0.93):
      print("\nAccuracy has reached 93%")
      self.model.stop_training = True
      
callbacks = myCallback()

"""# Training model"""

history = model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks=[callbacks], epochs=50)

"""# Create plot"""

def show_plots(history):
    loss_vals = history['loss']
    val_loss_vals = history['val_loss']
    epochs = range(1, len(history['accuracy'])+1)
    
    f, ax = plt.subplots(nrows=1,ncols=2,figsize=(15,6))
    
    # plot loss
    ax[0].plot(epochs, loss_vals, color='red',marker='o', linestyle=' ', label='Training Loss')
    ax[0].plot(epochs, val_loss_vals, color='black', marker='*', label='Validation Loss')
    ax[0].set_title('Training & Validation Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend(loc='best')
    ax[0].grid(True)
    
    # plot accuracy
    acc_vals = history['accuracy']
    val_acc_vals = history['val_accuracy']

    ax[1].plot(epochs, acc_vals, color='red', marker='o', ls=' ', label='Training Accuracy')
    ax[1].plot(epochs, val_acc_vals, color='black', marker='*', label='Validation Accuracy')
    ax[1].set_title('Training & Validation Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend(loc='best')
    ax[1].grid(True)
    
    plt.show()
    plt.close()
    
    # delete locals from heap before exiting
    del loss_vals, val_loss_vals, epochs, acc_vals, val_acc_vals
show_plots(history.history)

"""# Saving model into TF-Lite format"""

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

#save model in .tflite format
with tf.io.gfile.GFile('model.tflite', 'wb') as f:
  f.write(tflite_model)
