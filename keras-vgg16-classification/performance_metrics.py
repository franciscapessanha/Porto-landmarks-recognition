#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 19:57:23 2019

@author: mariafranciscapessanha
"""
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import models
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from keras.applications import vgg16

import glob
import os


model = load_model('small_last4.h5')


curr_path = os.getcwd()
set_path = os.path.normpath(os.path.join(curr_path, "../dataset/vgg16_resized_sets/test"))
classes = os.listdir(set_path)

predictions = []
ground_truth = []
for label in ['control']:
    print(label)
    image_ids = [f for f in glob.glob(os.path.join(set_path, '%s/*.jpg' % (label)))]
    for image_id in image_ids:
        image = load_img(image_id, target_size = (224,224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = vgg16.preprocess_input(image)
        preds = model.predict(image)
        print('Predicted:', decode_predictions(preds))
        

