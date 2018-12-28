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
        if file.endswith('.jpg') or file.endswith('.png'):
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
    #INSERIR MENSAGEM DE ERRO CASO NAO SEJA ENCONTRADA UMA ESCALA ADEQUADA

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
Get Normalization Parameters (mean and std)
==============================================================================
Calculates the mean and std of the training set in order to normalize all the 
samples.

Arguments:
    * train: train dataset
Returns:
    * mean: mean intensity for each channel of the training set
    * std: standard deviation of the intensity for each channel of the training set
"""   

def get_norm_parameters(train):
    blue_train = []
    green_train = []
    red_train = []
    for i in range(len(train)):
        b,g,r = cv.split(train[i])
        blue_train.append(b)
        green_train.append(g)
        red_train.append(r)
        
    mean_blue = np.mean(blue_train)
    mean_green = np.mean(green_train)
    mean_red = np.mean(red_train)
    
    std_blue = np.std(np.hstack(blue_train))
    std_green = np.std(np.hstack(green_train))
    std_red = np.std(np.hstack(red_train))
    
    return [mean_blue, mean_green, mean_red], [std_blue,std_green,std_red]

"""
Data normalization
==============================================================================
Normalizes a dataset according to the normalization parameters given (mean and std)

Arguments:
    * set_: image dataset
    * mean: mean intensity for each channel of the training set
    * std: standard deviation of the intensity for each channel of the training set
Returns:
    * set_: normalized dataset
"""   

def normalize_data(set_, mean, std):
    blue = []
    green = []
    red = []
    
    for i in range(len(set_)):
        b,g,r = cv.split(set_[i])
        blue.append(b)
        green.append(g)
        red.append(r)
    
    blue = [(b - mean[0])/std[0] for b in blue]
    green = [(g - mean[1])/std[1] for g in green]
    red = [(r - mean[2])/std[2] for r in red]

    set_ = [cv.merge((b,g,r)) for b,g,r in zip(blue,green,red)]
    
    return set_

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
Returns datasets after resizing and data normalization.

Returns:
    * x_train: list with the images of the train set
    * y_train: list with the labels of the train set
    * x_val: list with the images of the validation set
    * y_val: list with the labels of the validation set
    * x_test: list with the images of the test set
    * y_test: list with the labels of the test set

"""            
def get_data():
    curr_path = os.getcwd()
    
    arrabida = find_images(os.path.join(curr_path, 'dataset/images/arrabida'))
    camara = find_images(os.path.join(curr_path, 'dataset/images/camara'))
    clerigos = find_images(os.path.join(curr_path, 'dataset/images/clerigos'))
    musica = find_images(os.path.join(curr_path, 'dataset/images/musica'))
    serralves = find_images(os.path.join(curr_path, 'dataset/images/serralves'))
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
            
    """
    mean, std = get_norm_parameters(train)
    x_train = normalize_data(train, mean, std)
    x_val = normalize_data(val, mean, std) 
    x_test = normalize_data(test, mean, std)
    """
    
    save_data(train, y_train, "train")
    save_data(val,y_val,"val")
    save_data(test,y_test,"test")
    
    return x_train, y_train, x_val, y_val, x_test, y_test

x_train, y_train, x_val, y_val, x_test, y_test = get_data()


