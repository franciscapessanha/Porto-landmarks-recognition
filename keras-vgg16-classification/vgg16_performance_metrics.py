from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
from keras.applications import vgg16
from sklearn.metrics import confusion_matrix
import glob
import os


model = load_model('../models/vgg16.h5')
curr_path = os.getcwd()
set_path = os.path.normpath(os.path.join(curr_path, "../dataset/vgg16_resized_sets/test"))
classes =  ['arrabida', 'camara', 'clerigos', 'musica', 'serralves', 'control']

predictions = []
ground_truth = []
print(classes)

"""
Classes labels
===============
#arrabida = 0
#camara = 1
#clerigos = 2
#musica = 3
#serralves = 4
"""

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
precision = TP/(TP + FP)
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

