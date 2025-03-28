#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries

import tensorflow as tf  # TensorFlow library for deep learning
from tensorflow import keras  # Keras, a high-level API for building and training models
import numpy as np  # NumPy for numerical computations
import matplotlib.pyplot as plt  # Matplotlib for plotting data
import pandas as pd

# Import Keras model and layers
from keras.models import Sequential  # Sequential model allows layers to be added one by one
from keras.layers import Dense  # Fully connected layer (Dense)
from keras.layers import Dropout  # Dropout layer for regularization to prevent overfitting
from keras.layers import Flatten  # Flatten layer to convert 2D to 1D for the fully connected layer
from keras.layers import Activation  # Activation function like ReLU, sigmoid, etc.

# Import convolutional layers
from keras.layers import Conv2D  # 2D convolution layer to detect features like edges
from keras.layers import MaxPooling2D  # MaxPooling layer for downsampling (reducing dimensionality)

# Print TensorFlow version to ensure correct setup
print(tf.__version__)

# Import additional utilities
#from keras.optimizers import SGD  # Stochastic Gradient Descent optimizer
#from keras.preprocessing.image import ImageDataGenerator  # For data augmentation to generate new images by transformation
# Import additional utilities
from tensorflow.keras.optimizers import SGD  # Stochastic Gradient Descent optimizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # For data augmentation to generate new images by transformation


# In[2]:


pip install split-folders


# In[3]:


import pathlib
import splitfolders
import os
import shutil


# In[4]:


labels_df = pd.read_csv('metadata.csv')
labels_df.head()


# In[5]:


import pathlib
import splitfolders
import os
import shutil

# Dataset Path
in_data_dir = '/Users/pallavi/Desktop/Brain Tumor Data Set'
in_data_dir = pathlib.Path(in_data_dir)
print(os.listdir(in_data_dir))


# In[6]:


from PIL import Image
import os

def convert_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.tif')):
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)

            # Convert to RGB
            img = img.convert('RGB')

            # Save the image in the desired format (e.g., JPEG)
            img.save(os.path.join(output_dir, filename), 'JPEG')

# Example usage:
tumor_input_dir = '/Users/pallavi/Desktop/Brain Tumor Data Set/Brain Tumor'  # Path to tumor images
healthy_input_dir = '/Users/pallavi/Desktop/Brain Tumor Data Set/Healthy'  # Path to healthy images
tumor_output_dir = '/Users/pallavi/Desktop/project/standardized/tumor'  # Path to save converted tumor images
healthy_output_dir = '/Users/pallavi/Desktop/Project/standardized/healthy'  # Path to save converted healthy images

# Convert tumor images
convert_images(tumor_input_dir, tumor_output_dir)

# Convert healthy images
convert_images(healthy_input_dir, healthy_output_dir)


# In[7]:


import os

def count_images(directory):
    """Counts the number of images in a given directory."""
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.tif')):
            count += 1
    return count

# Paths to your original and standardized directories
tumor_input_dir = '/Users/pallavi/Desktop/Brain Tumor Data Set/Brain Tumor'  # Path to tumor images
healthy_input_dir = '/Users/pallavi/Desktop/Brain Tumor Data Set/Healthy'  # Path to healthy images
tumor_output_dir = '/Users/pallavi/Desktop/project/standardized/tumor'  # Path to save converted tumor images
healthy_output_dir = '/Users/pallavi/Desktop/Project/standardized/healthy'  # Path to save converted healthy images

# Count images in original directories
original_tumor_count = count_images(tumor_input_dir)
original_healthy_count = count_images(healthy_input_dir)

# Count images in standardized directories
standardized_tumor_count = count_images(tumor_output_dir)
standardized_healthy_count = count_images(healthy_output_dir)

# Print the results
print(f"Original Tumor Images: {original_tumor_count}")
print(f"Original Healthy Images: {original_healthy_count}")
print(f"Standardized Tumor Images: {standardized_tumor_count}")
print(f"Standardized Healthy Images: {standardized_healthy_count}")


# In[8]:


# Dataset Path
in_data_dir = '/Users/pallavi/Desktop/project/standardized'
in_data_dir = pathlib.Path(in_data_dir)
print(os.listdir(in_data_dir))


# In[9]:


# Specify the full path for the output directory
output_dir = '/Users/pallavi/Desktop/project/new_split'

# Remove the existing output directory if it exists
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)  # Deletes the existing directory and its contents


# In[10]:


# Splitting dataset into train and validation sets (80% train, 20% validation)
splitfolders.ratio(in_data_dir, output=output_dir, seed=20, ratio=(0.8, 0.2))


# ### New dataset path after splitting
# new_data_dir = pathlib.Path(output_dir)
# 
# ### Check and print the directory structure
# print("Train directory contents:", list((new_data_dir / 'train').glob('*/*')))  # Print class-wise images in train
# print("Validation directory contents:", list((new_data_dir / 'val').glob('*/*')))  # Print class-wise images in val'''
# 

# In[11]:


# New dataset path after splitting
new_data_dir = '/Users/pallavi/Desktop/project/new_split'
new_data_dir = pathlib.Path(new_data_dir)
print(os.listdir(new_data_dir))


# In[12]:


import pathlib
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset paths
train_dir = '/Users/pallavi/Desktop/project/new_split/train'
val_dir = '/Users/pallavi/Desktop/project/new_split/val'

# Create an ImageDataGenerator for augmentation and normalization for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    rotation_range=10,  # Small rotations for augmentation
    width_shift_range=0.1,  # Horizontal shifts
    height_shift_range=0.1,  # Vertical shifts
    zoom_range=0.1,  # Zoom in and out
    #brightness_range=[0.8, 1.2],  # Adjust brightness slightly
)

# Create an ImageDataGenerator for normalization for validation data (without augmentation)
val_datagen = ImageDataGenerator(
    rescale=1./255  # Normalize pixel values to [0, 1]
)

# Load and preprocess training images from directory
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Resize images to (224, 224) for CNN input
    batch_size=16,
    color_mode='grayscale',  # Use 'grayscale' if your images are single channel
    class_mode='binary',  # Use 'binary' for two classes (healthy and tumor)
    shuffle=True  # Enable shuffling for training data
)

# Load and preprocess validation images from directory (without augmentation)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=16,
    color_mode='grayscale',  # Use 'grayscale' if images are single channel
    class_mode='binary',
     shuffle=False  # Do not shuffle validation data for consistent evaluation
)

# Fitting the Data Generator to the training images (to calculate mean and std)
#train_datagen.fit(train_generator[0][0])  # Fit based on a batch of training images for mean/std

# Checking the shape of images in the train and validation generators
print(f"Train images shape: {train_generator[0][0].shape}")
print(f"Validation images shape: {val_generator[0][0].shape}")

print("Preprocessing completed.")


# ## VGG Model - Pending Code to execute

# In[13]:


import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D

# Define the model using VGG16
def create_vgg16_model(input_shape=(224, 224, 1), num_classes=2, learning_rate=1e-4):
    inputs = Input(shape=input_shape)

    # Convert grayscale to RGB (VGG16 requires 3 input channels)
    x = Conv2D(3, (1, 1), padding='same', activation='relu')(inputs)

    # Load VGG16 base model
    vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    vgg_base.trainable = False  # Freeze base layers initially

    # Add custom layers on top of VGG16
    x = vgg_base(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)  # Binary classification (e.g., Tumor or Healthy)

    vgg_model = Model(inputs, outputs)

    # Compile the model
    vgg_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return vgg_model

# Create the VGG16-based model
vgg_model = create_vgg16_model()

# Print the model summary
vgg_model.summary()


# In[14]:


# Set batch size
batch_size = 16

steps_per_epoch = len(train_generator.filenames) // batch_size
validation_steps = len(val_generator.filenames) // batch_size

print(f"Steps per epoch: {steps_per_epoch}, Validation steps: {validation_steps}")


# In[15]:


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import Callback
import math


# Early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,  # Stop training if validation loss doesn't improve for 3 epochs
    restore_best_weights=True  # Restore the best weights at the end of training
)

# Model checkpoint
checkpoint = ModelCheckpoint(
    filepath='bestCNNreg.keras',  # Save the best model
    monitor='val_loss',  # Monitor validation loss
    save_best_only=True,  # Save only when val_loss improves
    mode='min',  # Minimize val_loss
    verbose=1  # Print saving info
)

class DebugCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1} ended with logs: {logs}")

debug_callback = DebugCallback()

# Train the Model
history_vgg_model = vgg_model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=50,  # Number of epochs
    validation_data=val_generator,
    validation_steps=validation_steps,
    callbacks=[early_stopping, checkpoint, debug_callback],  # Add callbacks
    verbose=1 # Print training progress   
)


# In[ ]:




