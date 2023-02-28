# -*- coding: utf-8 -*-
"""
Created on Thu May 26 13:11:56 2022

@author: MSI
"""

import gc
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D,Flatten, Dense, Dropout
import tensorflow as tf
from tensorflow.keras import backend as K
import pickle

IMG_WIDTH,IMG_HEIGHT = 150,150
TRAIN_DATA_DIR = 'train'
VALIDATION_DATA_DIR = 'validation'
NB_TRAIN_SAMPLE = 4750
NB_VALIDATION_SAMPLES = 4750
EPOCHS = 50
BATCH_SIZE = 5

from keras.utils import np_utils
from sklearn import preprocessing
import matplotlib.pyplot as plt

# trainLabel = []
# le = preprocessing.LabelEncoder()
# le.fit(trainLabel[0])
# print("Classes: " + str(le.classes_))
# encodeTrainLabels = le.transform(trainLabel[0])

def build_model():
    if K.image_data_format()=='channels_first':
        input_shape = (3, IMG_WIDTH,IMG_HEIGHT)
    else:
        input_shape = (IMG_WIDTH,IMG_HEIGHT,3)
        
    model = Sequential();
    model.add(Conv2D(32,(3,3),input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(32,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(12))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def train_model(model):
    train_datagen = ImageDataGenerator(
                    
                    rescale=1.0/255,
                    shear_range = 0.2,
                    zoom_range=0/2,
                    horizontal_flip=True)
   
    test_datagen = ImageDataGenerator(rescale=1. /255)
    
    train_generator = train_datagen.flow_from_directory(
                        TRAIN_DATA_DIR,
                        target_size=(IMG_WIDTH,IMG_HEIGHT),
                        batch_size=BATCH_SIZE,
                        class_mode='categorical')
    
    validation_generator = test_datagen.flow_from_directory(
                        VALIDATION_DATA_DIR,
                        target_size=(IMG_WIDTH,IMG_HEIGHT),
                        batch_size=BATCH_SIZE,
                        class_mode='categorical')
    
    model.fit_generator(
                train_generator,
                steps_per_epoch=NB_TRAIN_SAMPLE // BATCH_SIZE,
                epochs = EPOCHS,
                validation_data = validation_generator,
                validation_steps = NB_TRAIN_SAMPLE // BATCH_SIZE)
    return model

def save_model(model):
    model.save('saved_model.h5')
    
def loadModel(nmModel):
    f = open(nmModel, 'rb')
    model = pickle.load(f)
    return model

# import numpy as np

def main():
    myModel = None
    tf.keras.backend.clear_session()
    gc.collect()
    myModel = build_model()
    myModel = train_model(myModel)
    save_model(myModel)
    # sample_image = "fish05.jpeg"
    # img = load_img(sample_image,target_size=(150,150))
    # x = img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    
    #x = x.reshape(3,150,150,-1)
    #print(myModel.predict(x))
    # result_predict=myModel.predict(x)
    # result_proba = np.argmax(result_predict, axis=-1)
    # print(result_predict)
    # print(result_proba)
    
main()
