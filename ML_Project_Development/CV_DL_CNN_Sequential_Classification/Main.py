import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import os
import numpy as np
import pandas as pd
import scipy
import sklearn
import keras
from keras.models import Sequential
import cv2
from skimage import io
%matplotlib inline

#Defining the File Path

cat=os.listdir("/Users/divyamereddy/Documents/GitHub/MachineLearningProjects/ML_Project_Development/CV_DL_CNN_Classification/kagglecatsanddogs_5340/PetImages/Cat/")
dog=os.listdir("/Users/divyamereddy/Documents/GitHub/MachineLearningProjects/ML_Project_Development/CV_DL_CNN_Classification/kagglecatsanddogs_5340/PetImages/Dog")
filepath="/Users/divyamereddy/Documents/GitHub/MachineLearningProjects/ML_Project_Development/CV_DL_CNN_Classification/kagglecatsanddogs_5340/PetImages/Cat/"
filepath2="/Users/divyamereddy/Documents/GitHub/MachineLearningProjects/ML_Project_Development/CV_DL_CNN_Classification/kagglecatsanddogs_5340/PetImages/Dog/"

#Loading the Images
import imageio
import pillow
images=[]
label = []
for i in cat:
    if('.jpg' in i ):
        image =  imageio.imread(filepath+i)
        images.append(image)
        label.append(0) #for cat images

for i in dog:
    image = scipy.misc.imread(filepath2+i)
    images.append(image)
    label.append(1) #for dog images

#resizing all the images

for i in range(0,23000):
    images[i]=cv2.resize(images[i],(300,300))

#converting images to arrays

images=np.array(images)
label=np.array(label)

# Defining the hyperparameters

filters=10
filtersize=(5,5)

epochs =5
batchsize=128

input_shape=(300,300,3)

#Converting the target variable to the required size

from keras.utils.np_utils import to_categorical
label = to_categorical(label)

#Defining the model

model = Sequential()

model.add(keras.layers.InputLayer(input_shape=input_shape))

model.add(keras.layers.convolutional.Conv2D(filters, filtersize, strides=(1, 1), padding='valid', data_format="channels_last", activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(units=2, input_dim=50,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(images, label, epochs=epochs, batch_size=batchsize,validation_split=0.3)

model.summary()
