## Standard Python Libraries ##
import os, sys
import numpy as np

## OpenCV Libraries ##
import cv2, glob

## Keras Libraries ##
from tensorflow import keras
from tensorflow.keras.preprocessing.image import image_dataset_from_directory
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

## Dataset Preprocessing ##
width  = 550
height = 550
image_size = (width,height)
batch_size = 32

images = glob.glob("*.jpg")
for image in images:
    img = cv2.imread(image, 0)
    resized_img = cv2.resize(image,image_size,interpolation = cv2.INTER_AREA)
    cv2.imwrite(img,resized_img)

## Generate Datasets ##
Dataset_Dir     = os.path.join("100_Leaves","data")
Train_DS        = image_dataset_from_directory(directory=Dataset_Dir,validation_split=0.25,subset="training"  ,image_size=image_size,batch_size=batch_size)
Validation_DS   = image_dataset_from_directory(directory=Dataset_Dir,validation_split=0.25,subset="validation",image_size=image_size,batch_size=batch_size)

## Augment Dataset - Create new data by modifying available data ## 
Augmented_Data  = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1)),
    ]
) # Randomly flip & rotate images in the preprocessing layer.

Train_DS        = Train_DS.prefetch(buffer_size = 32)
Validation_DS   = Validation_DS.prefetch(buffer_size = 32) # Prefetch data into a buffer so that it's immediately available to the hardware when the previous batch is complete.

## Build Model ##
model = Sequential()


## Train Model ##