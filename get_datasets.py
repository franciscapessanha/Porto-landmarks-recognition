import os
import numpy as np
import cv2 as cv
from sklearn.model_selection import train_test_split

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
        if file.endswith('.jpg'):
            files += [file]
            full_path += [os.path.join(directory,file)]    
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
Save Data
==============================================================================
Saves the images in folders, organized by dataset and class

Arguments:
    * set_: image dataset
    * y: image labels
    * folder_name: name of the dataset
"""
def save_data(set_, y, folder_name):
    d0 = 0
    d1 = 0
    d2 = 0
    d3 = 0
    d4 = 0
        
    for image, label in zip(set_,y):
        curr_path = os.getcwd()
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        if label == 0: #arrabida
            filename = (curr_path + "/" + folder_name + "/arrabida/arrabida_%d.jpg") % d0
            cv.imwrite(filename, image)
            d0 += 1
        elif label == 1: #camara
            filename = (curr_path + "/" + folder_name + "/camara/camara_%d.jpg") % d1
            cv.imwrite(filename, image)
            d1 += 1
        elif label == 2: #clerigos
            filename = (curr_path + "/" + folder_name + "/clerigos/clerigos_%d.jpg") % d2
            cv.imwrite(filename, image)
            d2 += 1
        elif label == 3: #musica
            filename = (curr_path + "/" + folder_name + "/musica/musica_%d.jpg") % d3
            cv.imwrite(filename, image)
            d3 += 1
        elif label == 4: #serralves
            filename = (curr_path + "/" + folder_name + "/serralves/serralves_%d.jpg") % d4
            cv.imwrite(filename, image)
            d4 += 1
            
"""
Get Data
==============================================================================
Saves datasets after resizing.

"""            
def run():
    curr_path = os.getcwd()
    
    arrabida = find_images(os.path.join(curr_path, 'images/arrabida'))
    camara = find_images(os.path.join(curr_path, 'images/camara'))
    clerigos = find_images(os.path.join(curr_path, 'images/clerigos'))
    musica = find_images(os.path.join(curr_path, 'images/musica'))
    serralves = find_images(os.path.join(curr_path, 'images/serralves'))
    x = np.concatenate((arrabida, camara, clerigos, musica, serralves))
    
    y_arrabida = np.zeros(np.shape(arrabida),np.uint8)
    y_camara = np.ones(np.shape(camara),np.uint8)
    y_clerigos = np.ones(np.shape(clerigos),np.uint8)*2
    y_musica = np.ones(np.shape(musica),np.uint8)*3
    y_serralves = np.ones(np.shape(serralves),np.uint8)*4
    y = np.concatenate((y_arrabida, y_camara, y_clerigos, y_musica, y_serralves))
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x,y)
    
    train = []
    val = []
    test = []
    print("train")
    for path in x_train:
        image = cv.imread(path[0])
        image = resize_images(image)
        train.append(image)
    print("val")
    for path in x_val:
        image = cv.imread(path[0])
        image = resize_images(image)
        val.append(image)
    print("test")
    for path in x_test:
        image = cv.imread(path[0])
        image = resize_images(image)
        test.append(image)
            
    save_data(train, y_train, "train")
    save_data(val,y_val,"val")
    save_data(test,y_test,"test")

run()


