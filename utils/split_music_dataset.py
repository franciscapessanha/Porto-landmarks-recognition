import os
import numpy as np
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
    * validation set (15%)
    * test set (15%)
    
Arguments:
    * x: images paths
    * y: images labels

Returns:
    * x_train: list with the images of the train set
    * y_train: list with the labels of the train set
    * x_val: list with the images of the validation set
    * y_val: list with the labels of the validation set
    * x_test: list with the images of the test set
    * y_test: list with the labels of the test set

"""
def split_data(x, y):
    x_train, xi_test, y_train, yi_test = train_test_split(x, y, stratify=y, test_size=0.3)
    x_test, x_val, y_test, y_val = train_test_split(xi_test, yi_test, stratify=yi_test, test_size=0.5)
    
    return x_train, y_train, x_val, y_val, x_test, y_test
    
"""
Save Copy
==============================================================================
Saves the images organized by dataset and class

Arguments:
    * path: path to the image
    * label: image class (frente, tras, lado, musica, serralves or outro)
    * folder_name: dataset (train,test or val)
"""
def save_copy(path, label, folder_name):

    curr_path = os.getcwd()
    path_folder = os.path.normpath(os.path.join(curr_path, "../dataset/tier3-musica/divided_sets", folder_name, label))

    if not os.path.exists(path_folder):
        os.makedirs(path_folder)
    filename = path_folder + path[path.rindex('/'):]
    shutil.copy2(path,filename)

"""
Run
==============================================================================

"""            
def run():
    curr_path = os.getcwd()
    
    print("1. Get images paths")
    frente = find_images(os.path.normpath(os.path.join(curr_path, '../dataset/tier3-musica/frente')))
    tras = find_images(os.path.normpath(os.path.join(curr_path, '../dataset/tier3-musica/tras')))
    lado = find_images(os.path.normpath(os.path.join(curr_path, '../dataset/tier3-musica/lado')))
    outro = find_images(os.path.normpath(os.path.join(curr_path, '../dataset/tier3-musica/outro')))
    x = np.concatenate((frente, tras, lado))
    
    y_frente = ['frente' for x in range(len(frente))]
    y_tras = ['tras' for x in range(len(tras))]
    y_lado = ['lado' for x in range(len(lado))]
    y = np.concatenate((y_frente, y_tras, y_lado))
    
    print("2. Split into datasets (train, test, validation)")
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x,y)
    
    print("3. Save original images by dataset")
    divided_sets_folder = os.path.normpath(os.path.join(curr_path, "../dataset/tier3-musica/divided_sets"))
    if os.path.exists(divided_sets_folder):
        shutil.rmtree(divided_sets_folder)
    
    print("  3.1 Train")
    for path,i in zip(x_train, range(len(x_train))):
        save_copy(path[0], y_train[i], "trainval")
       
    print("  3.2 Validation")
    for path,i in zip(x_val, range(len(x_val))):
        save_copy(path[0], y_val[i], "trainval")
    
    print("  3.3 Test")
    for path,i in zip(x_test, range(len(x_test))):
        save_copy(path[0], y_test[i], "test")
    
    for path in outro:
        save_copy(path[0], 'outro', "test")
    
run()

