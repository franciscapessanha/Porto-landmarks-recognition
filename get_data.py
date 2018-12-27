import os
import numpy as np
import cv2 as cv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

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
    * mode: default or cross_val (cross validation with 5 folds)

Returns:
    if mode = "default":
        * x_train: list with the images of the train set
        * y_train: list with the labels of the train set
        * x_val: list with the images of the validation set
        * y_val: list with the labels of the validation set
        * x_test: list with the images of the test set
        * y_test: list with the labels of the test set

    if mode = "cross_val":
        * cv_x_train: list with 5 different set of images of the train set
        * cv_y_train: list with 5 different set of labels of the train set
        * cv_x_val: list with 5 different set of images for the validation set
        * cv_y_val: list with 5 different set of labels of the validation set
        * x_test: list with the images of the test set
        * y_test: list with the labels of the test set
"""
def split_data(x, y, mode = "default"):
    x_train, xi_test, y_train, yi_test = train_test_split(x, y, stratify=y, test_size=0.3)
    x_test, x_val, y_test, y_val = train_test_split(xi_test, yi_test, stratify=yi_test, test_size=0.5)

    if mode == "default":
        return x_train, y_train, x_val, y_val, x_test, y_test
    
    elif mode == "cross_val": 
        x_cv = np.concatenate((x_train, x_val))
        y_cv = np.concatenate((y_train, y_val))
        
        skf = StratifiedKFold(n_splits = 5)
        
        cv_x_train = []
        cv_y_train = []
        cv_x_val = []
        cv_y_val = []
        
        for train_index, val_index in skf.split(x_cv, y_cv):
             x_train, x_val = x[train_index], x[val_index]
             y_train, y_val = y[train_index], y[val_index]
             
             cv_x_train.append(x_train)
             cv_y_train.append(y_train)
             cv_x_val.append(x_val)
             cv_y_val.append(y_val)
    
        return cv_x_train, cv_y_train,cv_x_val, cv_y_val, x_test, y_test

"""
Load Data
==============================================================================
Loads the images on the paths given

Arguments:
    * x: list of all the image paths
Returns:
    * loaded_x: list with all the images
"""

def load_data(x):
    loaded_x = []
    for i in range(len(x)):
        loaded_x.append(cv.imread(x[i][0]))   
    return loaded_x

"""
Resize Images
==============================================================================
Resizes all images to a defined size (h_size x w_size)

Arguments:
    * set_: image dataset
    * h_size: desired height
    * w_size: desired width
Returns:
    * resized_set : resized dataset
"""  
def resize_images(set_, h_size = 224, w_size = 224): #shape images ImageNet
    resized_set = []
    
    
    #INSERIR MENSAGEM DE ERRO CASO NAO SEJA ENCONTRADA UMA ESCALA ADEQUADA
    
    for image in set_:
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
            
        resized_set.append(cv.copyMakeBorder(image, top = b_top,bottom = b_bot, left = b_left, right = b_right, borderType = cv.BORDER_CONSTANT, value = [0,0,0]))   
    return resized_set

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
Get Data
==============================================================================
Returns datasets after resizing and data normalization.

Arguments:
    * mode: "default" or "cross-val"
Returns:
    if mode = "default":
        * x_train: list with the images of the train set
        * y_train: list with the labels of the train set
        * x_val: list with the images of the validation set
        * y_val: list with the labels of the validation set
        * x_test: list with the images of the test set
        * y_test: list with the labels of the test set

    if mode = "cross_val":
        * x_train: list with 5 different set of images of the train set
        * y_train: list with 5 different set of labels of the train set
        * x_val: list with 5 different set of images for the validation set
        * y_val: list with 5 different set of labels of the validation set
        * x_test: list with the images of the test set
        * y_test: list with the labels of the test set
"""
   
def get_data(mode = "default"):
    curr_path = os.getcwd()
    
    arrabida = find_images(os.path.join(curr_path, 'images/arrabida'))
    camara = find_images(os.path.join(curr_path, 'images/camara'))
    clerigos = find_images(os.path.join(curr_path, 'images/clerigos'))
    musica = find_images(os.path.join(curr_path, 'images/musica'))
    serralves = find_images(os.path.join(curr_path, 'images/serralves'))
    x = np.concatenate((arrabida, camara, clerigos, musica, serralves))
    
    y_arrabida = np.zeros(np.shape(arrabida))
    y_camara = np.ones(np.shape(camara))
    y_clerigos = np.ones(np.shape(clerigos))*2
    y_musica = np.ones(np.shape(musica))*3
    y_serralves = np.ones(np.shape(serralves))*4
    y = np.concatenate((y_arrabida, y_camara, y_clerigos, y_musica, y_serralves))
   
    if mode == "default":
        x_train, y_train, x_val, y_val, x_test, y_test = split_data(x,y)
        
        x_train = load_data(x_train)
        x_val = load_data(x_val)
        x_test = load_data(x_test)
        
        x_train = resize_images(x_train)
        x_val = resize_images(x_val)
        x_test = resize_images(x_test)
        
        mean, std = get_norm_parameters(x_train)
        x_train = normalize_data(x_train, mean, std)
        x_val = normalize_data(x_val, mean, std) 
        x_test = normalize_data(x_test, mean, std)
        
        
    elif mode == "cross_val":
        cv_x_train, cv_y_train,cv_x_val, cv_y_val, x_test, y_test =split_data(x,y, "cross_val")
        x_train = []
        x_val = []
        x_test = []
        
        x0_test = load_data(x_test)
        x0_test = resize_images(x_test)
        
        for x_train, x_val in zip(cv_x_train, cv_x_val):
            
            xi_train = load_data(x_train)
            xi_val = load_data(x_val)
            
            xi_train = resize_images(xi_train)
            xi_val = resize_images(xi_val)
            
            mean, std = get_norm_parameters(xi_train)
            xi_train = normalize_data(xi_train, mean, std)
            xi_val = normalize_data(xi_val, mean, std) 
            xi_test = normalize_data(x0_test, mean, std)
            
            x_train.append(xi_train)
            x_val.append(xi_val)
            x_test.append(xi_test)
            
    return x_train, y_train, x_val, y_val, x_test, y_test

x_train, y_train, x_val, y_val, x_test, y_test = get_data()


