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
Resize Image
==============================================================================
Resizes an image to a defined size (h_size x w_size)

Arguments:
    * image: image
    * h_size: desired height
    * w_size: desired width
Returns:
    * resized_set : resized dataset
"""  
def resize_images(image, h_size = 224, w_size = 224): #shape images ImageNet    

    h, w, _ = image.shape
    for scale in np.arange(2.0, 0.0, -0.005):
        if w * scale <= w_size and h * scale <= h_size:
            break
    image = cv.resize(image, (0,0), fx=scale, fy=scale) 
    new_h, new_w, h = image.shape
            
    if (h_size - new_h) % 2 == 0: # height is even
        b_top = int((h_size - new_h) /2)
        b_bot = b_top
        
    else:
        b_top = int((h_size - new_h) / 2)
        b_bot = int(h_size - new_h - b_top)
    
    if (w_size - new_w % 2) == 0: # width is even
        b_left = int((w_size - new_w) / 2)
        b_right = b_left
    else:
        b_left= int((w_size - new_w)/2)
        b_right = int((w_size - new_w) - b_left)
        
    resized_image = cv.copyMakeBorder(image, top = b_top,bottom = b_bot, left = b_left, right = b_right, borderType = cv.BORDER_CONSTANT, value = [0,0,0])   
    return resized_image

"""
Save Image
==============================================================================
Saves the images in folders, organized by dataset and class

Arguments:
    * image: image
    * label: image label
    * i: image index
    * folder_name: name of the dataset
"""
def save_image(path, image, label, folder_name):
    curr_path = os.getcwd()
    path_arrabida = os.path.normpath(os.path.join(curr_path, "../dataset/divided_sets/", folder_name, "arrabida"))
    path_camara = os.path.normpath(os.path.join(curr_path, "../dataset/divided_sets/", folder_name, "camara"))
    path_clerigos = os.path.normpath(os.path.join(curr_path, "../dataset/divided_sets/", folder_name, "clerigos"))
    path_musica = os.path.normpath(os.path.join(curr_path, "../dataset/divided_sets/", folder_name, "musica"))
    path_serralves = os.path.normpath(os.path.join(curr_path, "../dataset/divided_sets/", folder_name, "serralves"))

    if not os.path.exists(path_arrabida):
        os.makedirs(path_arrabida)
    
    if not os.path.exists(path_camara):
        os.makedirs(path_camara)
    
    if not os.path.exists(path_clerigos):
        os.makedirs(path_clerigos)
        
    if not os.path.exists(path_musica):
            os.makedirs(path_musica)

    if not os.path.exists(path_serralves):
            os.makedirs(path_serralves)

    if label == 0: #arrabida
        filename = path_arrabida + path[path.rindex('/'):]
        cv.imwrite(filename, image)
    elif label == 1: #camara
        filename = path_camara + path[path.rindex('/'):]
        cv.imwrite(filename, image)
    elif label == 2: #clerigos
        filename = path_clerigos + path[path.rindex('/'):]
        cv.imwrite(filename, image)
    elif label == 3: #musica
        filename = path_musica + path[path.rindex('/'):]
        cv.imwrite(filename, image)
    elif label == 4: #serralves
        filename = path_serralves + path[path.rindex('/'):]
        cv.imwrite(filename, image)

"""
Save Annotations
==============================================================================
Saves the annotation in folders, organized by dataset and class

Arguments:
    * annotation: image
    * label: image label
    * i: image index
    * folder_name: name of the dataset
"""
def save_copy(path, label, folder_name):

    curr_path = os.getcwd()
    path_arrabida = os.path.normpath(os.path.join(curr_path, "../dataset/divided_sets/", folder_name, "arrabida"))
    path_camara = os.path.normpath(os.path.join(curr_path, "../dataset/divided_sets/", folder_name, "camara"))
    path_clerigos = os.path.normpath(os.path.join(curr_path, "../dataset/divided_sets/", folder_name, "clerigos"))
    path_musica = os.path.normpath(os.path.join(curr_path, "../dataset/divided_sets/", folder_name, "musica"))
    path_serralves = os.path.normpath(os.path.join(curr_path, "../dataset/divided_sets/", folder_name, "serralves"))
    
    if not os.path.exists(path_arrabida):
        os.makedirs(path_arrabida)
    if not os.path.exists(path_camara):
        os.makedirs(path_camara)
    if not os.path.exists(path_clerigos):
        os.makedirs(path_clerigos)
    if not os.path.exists(path_musica):
        os.makedirs(path_musica)
    if not os.path.exists(path_serralves):
        os.makedirs(path_serralves)

    if label == 0: #arrabida
        filename = path_arrabida + path[path.rindex('/'):]
        shutil.copy2(path,filename)
    elif label == 1: #camara
        filename = path_camara + path[path.rindex('/'):]
        shutil.copy2(path,filename)
    elif label == 2: #clerigos
        filename = path_clerigos + path[path.rindex('/'):]
        shutil.copy2(path,filename)
    elif label == 3: #musica
        filename = path_musica + path[path.rindex('/'):]
        shutil.copy2(path,filename)
    elif label == 4: #serralves
        filename = path_serralves + path[path.rindex('/'):]
        shutil.copy2(path,filename)
"""
Get Data
==============================================================================
Saves datasets after resizing.

"""            

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

print("2. Split into datasets (train, test, validation)")
x_train, y_train, x_val, y_val, x_test, y_test = split_data(x,y)


print("3. Save resized images by dataset")
print("  3.1 Train")
for path,i in zip(x_train, range(len(x_train))):
    image = cv.imread(path[0])
    image = resize_images(image)
    save_image(path[0], image, y_train[i], "resized_train")
   
print("  3.2 Validation")
for path,i in zip(x_val, range(len(x_val))):
    image = cv.imread(path[0])
    image = resize_images(image)
    save_image(path[0], image, y_val[i], "resized_val")
   
print("  3.3 Test")
for path,i in zip(x_test, range(len(x_test))):
    image = cv.imread(path[0])
    image = resize_images(image)
    save_image(path[0], image, y_test[i], "resized_test")

print("4. Save original images by dataset")
print("  4.1 Train")
for path,i in zip(x_train, range(len(x_train))):
    if os.path.exists(ann_path):
        
        save_copy(ann_path, y_train[i], "annotations_train")
        save_copy(path[0], y_train[i], "original_train")
   
print("  4.2 Validation")
for path,i in zip(x_val, range(len(x_val))):
    if os.path.exists(ann_path):
        save_copy(path[0], y_val[i], "original_val")

print("  4.3 Test")
for path,i in zip(x_test, range(len(x_test))):
    if os.path.exists(ann_path):
        save_copy(path[0], y_test[i], "original_test")



