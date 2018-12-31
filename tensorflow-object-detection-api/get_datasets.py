import os
import numpy as np
import cv2 as cv
from sklearn.model_selection import train_test_split
import shutil

"""
Find Images
==============================================================================
Finds the images on the directory with the extension provided.

Arguments:
    * directory: path in which we want to find a file
Returns:
    * full paths: full path to each of the files
"""

def find_images(directory):
    files = []
    full_path = []
    for file in os.listdir(directory):
        if file.endswith('.jpg') or file.endswith('.png'):
            files += [file]
            full_path += [os.path.join(directory, file)]
        else:
            print("Images format must be '.jpg' or '.png'")
            
    files.sort()
    full_path.sort()
    return np.array(full_path).reshape(-1,1)

"""
Split data
==============================================================================
Split the data in:
    * training set (70%)
    * validation set (30%)
    
Arguments:
    * x: images paths
    * y: images labels

Returns:
    * x_train: list with the images of the train set
    * y_train: list with the labels of the train set
    * x_val: list with the images of the validation set
    * y_val: list with the labels of the validation set

"""
def split_data(x, y):
    x_train, x_val, y_train, y_val = train_test_split(x, y, stratify=y, test_size=0.3)

    return x_train, y_train, x_val, y_val

"""
Save Annotations
==============================================================================
Saves the annotation in folders, organized by dataset and class

Arguments:
    * annotation: image
    * folder_name: name of the dataset
"""
def save_copy(path, folder_name):

    new_path = os.path.normpath(os.path.join(os.getcwd(), "../dataset/divided_sets/", folder_name))

    if not os.path.exists(new_path):
        os.makedirs(new_path)

    filename = (new_path + path[path.rindex('/'):])
    shutil.copy2(path, filename)

"""
Get Data
==============================================================================
Saves datasets after resizing.

"""            
def run():
    curr_path = os.getcwd()

    print("1. Get images paths")
    arrabida = find_images(os.path.normpath(os.path.join(curr_path, '../dataset/images/arrabida')))
    camara = find_images(os.path.normpath(os.path.join(curr_path, '../dataset/images/camara')))
    clerigos = find_images(os.path.normpath(os.path.join(curr_path, '../dataset/images/clerigos')))
    musica = find_images(os.path.normpath(os.path.join(curr_path, '../dataset/images/musica')))
    serralves = find_images(os.path.normpath(os.path.join(curr_path, '../dataset/images/serralves')))
    x = np.concatenate((arrabida, camara, clerigos, musica, serralves))
    
    y_arrabida = np.zeros(np.shape(arrabida),np.uint8)
    y_camara = np.ones(np.shape(camara),np.uint8)
    y_clerigos = np.ones(np.shape(clerigos),np.uint8)*2
    y_musica = np.ones(np.shape(musica),np.uint8)*3
    y_serralves = np.ones(np.shape(serralves),np.uint8)*4
    y = np.concatenate((y_arrabida, y_camara, y_clerigos, y_musica, y_serralves))
    
    print("2. Split into datasets (train, test)")
    x_train, y_train, x_val, y_val = split_data(x,y)

    #ACRESCENTAR ANOTAÇÕES QUE FALTAM
    print("3. Save original images by dataset")
    print("  3.1 Train")
    for path in x_train:
        ann_path = path[0].replace("images", "annotations").replace("jpg", "xml")
        if os.path.exists(ann_path):
            save_copy(ann_path, "train")
            save_copy(path[0], "train")
   
    print("  3.2 Validation")
    for path in x_val:
        ann_path = path[0].replace("images", "annotations").replace("jpg", "xml")
        if os.path.exists(ann_path):
            save_copy(ann_path, "val")
            save_copy(path[0], "val")

run()