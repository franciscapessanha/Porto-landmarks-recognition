#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 17:01:54 2018

@author: mariafranciscapessanha
"""

import os
import numpy as np
import cv2 as cv
from sklearn.model_selection import train_test_split


def find_extension(directory):
    files = []
    full_path = []
    for file in os.listdir(directory):
        if file.endswith('.jpg') or file.endswith('.png'):
            files += [file]
            full_path += [os.path.join(directory,file)]
            
    files.sort()
    full_path.sort()
    return files, full_path


def get_paths(path):
    _, paths = find_extension(path)
    return np.array(paths).reshape(-1,1)

def load_data(x):
    new_x = []
    for i in range(len(x)):
        new_x.append(cv.imread(x[i][0]))   
    return new_x
     
def normalize_data(train, test):
    blue_train = []
    green_train = []
    red_train = []
    
    blue_test = []
    green_test = []
    red_test = []
    
    for i, j in zip(range(len(train)), range(len(test))):
        
        b,g,r = cv.split(train[i])
        blue_train.append(b)
        green_train.append(g)
        red_train.append(r)
        
        b_t,g_t,r_t = cv.split(test[i])
        blue_test.append(b_t)
        green_test.append(g_t)
        red_test.append(r_t)
     
    mean_blue = np.mean(blue_train)
    mean_green = np.mean(green_train)
    mean_red = np.mean(red_train)
    
    std_blue = np.std(np.hstack(blue_train))
    std_green = np.std(np.hstack(green_train))
    std_red = np.std(np.hstack(red_train))
    
    
    blue_train = (blue_train - mean_blue)/std_blue
    green_train = (green_train - mean_green)/std_green
    red_train = (red_train - mean_red)/std_red
    train = np.concatenate((blue_train,green_train,red_train))
    
    blue_test = (blue_test - mean_blue)/std_blue
    green_test = (green_test - mean_green)/std_green
    red_test = (red_test - mean_red)/std_red
    test = np.concatenate((blue_test,green_test,red_test))

    return train, test

def resize_images(set):
    #definir!!!
    return set
    
#def get_data():
curr_path = os.getcwd()

arrabida = get_paths(os.path.join(curr_path, 'images/arrabida'))
camara = get_paths(os.path.join(curr_path, 'images/camara'))
clerigos = get_paths(os.path.join(curr_path, 'images/clerigos'))
musica = get_paths(os.path.join(curr_path, 'images/musica'))
serralves = get_paths(os.path.join(curr_path, 'images/serralves'))

y_arrabida = np.zeros(np.shape(arrabida))
y_camara = np.ones(np.shape(camara))
y_clerigos = np.ones(np.shape(clerigos))*2
y_musica = np.ones(np.shape(musica))*3
y_serralves = np.ones(np.shape(serralves))*4

#divide arrabida set
x_train_arr, x_test_arr, y_train_arr, y_test_arr = train_test_split(arrabida, y_arrabida, test_size = 0.3, random_state=0)
#divide camara set
x_train_cam, x_test_cam, y_train_cam, y_test_cam = train_test_split(camara, y_camara, test_size = 0.3, random_state=0)
#divide clerigos set
x_train_cle, x_test_cle, y_train_cle, y_test_cle = train_test_split(clerigos, y_clerigos, test_size = 0.3, random_state=0)
#divide musica set
x_train_mu, x_test_mu, y_train_mu, y_test_mu = train_test_split(musica, y_musica, test_size = 0.3, random_state=0)
#divide serralves set
x_train_serr, x_test_serr, y_train_serr, y_test_serr = train_test_split(serralves, y_serralves, test_size = 0.3, random_state=0)

x_train = np.concatenate((x_train_arr, x_train_cam, x_train_cle, x_train_mu, x_train_serr))
x_test = np.concatenate((x_test_arr, x_test_cam, x_test_cle, x_test_mu, x_test_serr))
y_train = np.concatenate((y_train_arr, y_train_cam, y_train_cle, y_train_mu, y_train_serr))
y_test = np.concatenate((y_test_arr, y_test_cam, y_test_cle, y_test_mu, y_test_serr))

x_train = load_data(x_train)
x_test = load_data(x_test) 

"""
INSERIR RESIZE
"""
x_train, x_test = normalize_data(x_train, x_test)
    
#return x_train, y_train, x_test, y_test

#x_train, y_train, x_test, y_test = get_data()