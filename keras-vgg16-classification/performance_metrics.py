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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
import glob
import os
"""
Confusion Matrix
===============================================================================

Calculates the confusion matrix given a prediction and a label

Arguments:
    * predictions: results obtained from the classifier
    * labels: ground truth of the classification
        
Return:
    * true_positives
    * false_negatives
    * false_positives
    * true_negatives
    
"""

def confusionMatrix(predictions, labels, true):
    true_positives = 0
    false_negatives = 0
    false_positives = 0
    true_negatives = 0
    
    for i in range(len(predictions)):
        if predictions[i] == labels[i] :
            if  predictions[i] == true:
                true_positives += 1
            else: 
                true_negatives += 1
        elif predictions[i] != labels[i]:
            if predictions[i] == 1.0:
                false_positives += 1
            elif predictions[i] == 0.0:
                false_negatives += 1
                
    return np.asarray([[true_positives, false_negatives], [false_positives, true_negatives]]) 
    
"""
getPerformanceMetrics
=========================

Calculates accuracy, precision, recall, auc for evaluation given an array of predictions and the corresponding ground truth

Arguments: 
    * predictions- array with predicted results
    * labels-  corresponding ground true

Return: 
    * accuracy 
    * precision 
    * recall 
    * auc 
"""


def getPerformanceMetrics(predictions, labels):
    tn, fp, fn, tp = confusion_matrix(labels, predictions)
    
    true_positives = c_matrix[0,0]
    false_negatives = c_matrix[0,1]
    false_positives = c_matrix[1,0]
    true_negatives = c_matrix[1,1]

    accuracy = (true_positives + true_negatives)/(true_positives + true_negatives + false_positives + false_negatives)
    precision = (true_positives)/(true_positives + false_positives + 10**-12)
    
    recall = (true_positives)/(true_positives + false_negatives)
    #matrix = np.asarray([[true_positives, false_negatives], [false_positives, true_negatives]])
    
    fp_rate, tp_rate, thresholds = metrics.roc_curve(labels, predictions, pos_label = 1)
    auc = metrics.auc(fp_rate, tp_rate)
    
    return accuracy, precision, recall, auc
        

model = load_model('small_last4.h5')


curr_path = os.getcwd()
set_path = os.path.normpath(os.path.join(curr_path, "../dataset/vgg16_resized_sets/test"))
classes =  ['arrabida', 'camara', 'clerigos', 'musica', 'serralves', 'control']

predictions = []
ground_truth = []
print(classes)

#arrabida = 0
#camara = 1
#clerigos = 2
#musica = 3
#serralves = 4

predictions = []
ground_truth = []
for label,i in zip(classes, range(len(classes))):
    print(label)
    image_ids = [f for f in glob.glob(os.path.join(set_path, '%s/*.jpg' % (label)))]
    ground_truth.append(np.ones(np.shape(image_ids))*i)
    for image_id in image_ids:
        image = load_img(image_id, target_size = (224,224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = vgg16.preprocess_input(image)
        predictions.append(model.predict_classes(image)[0])
        #print('Predicted:', preds)

ground_truth = np.hstack(ground_truth)
ground_truth = [int(gt) for gt in ground_truth]

c_matrix = confusion_matrix(ground_truth, predictions)
print(c_matrix)
FP = c_matrix.sum(axis=0) - np.diag(c_matrix)  
FN = c_matrix.sum(axis=1) - np.diag(c_matrix)
TP = np.diag(c_matrix)
TN = sum(c_matrix.ravel())- (FP + FN + TP)

accuracy = []
precision = []
recall = []

accuracy = (TP + TN)/(TP + TN + FP + FN)
precision = TP/(TP + FP + 10**-12)
recall = TP/(TP + FN)

acc_global = (sum(TP) + sum(TN))/(sum(TP)  + sum(TN)  + sum(FP)  + sum(FN))
precision_global = sum(TP) /(sum(TP)  + sum(FP))
recall_global = sum(TP) /(sum(TP) + sum(FN))


for i in range(len(classes)):
    print(classes[i])
    print("=================")
    print("Accuracy = %.2f \nPrecision = %.2f \nRecall = %.2f \n" % (accuracy[i],precision[i], recall[i]))

print("Global Metrics\n=================")
print("Accuracy = %.2f \nPrecision = %.2f \nRecall = %.2f \n" % (acc_global,precision_global, recall_global))

