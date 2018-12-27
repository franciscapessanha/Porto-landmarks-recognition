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
import matplotlib.pyplot as plt


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
    train = x_train
    test = x_test
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
    
    
    blue_train = [(blue - mean_blue)/std_blue for blue in blue_train]
    green_train = [(green - mean_green)/std_green for green in green_train]
    red_train = [(red - mean_red)/std_red for red in red_train]
    train = [np.concatenate((b,g,r)) for b,g,r in zip(blue_train,green_train,red_train)]
    
    blue_test = [(blue - mean_blue)/std_blue for blue in blue_test]
    green_test = [(green - mean_green)/std_green for green in green_test]
    red_test = [(red - mean_red)/std_red for red in red_test]
    test = [np.concatenate((b,g,r)) for b,g,r in zip(blue_test,green_test,red_test)]

    return train, test

def resize_images(set_, h_size = 50, w_size = 100):
    resized = []
    
    """
    INSERIR MENSAGEM DE ERRO CASO NAO SEJA ENCONTRADA UMA ESCALA ADEQUADA
    """
    for image in set_:
        h, w, _ = image.shape
        for scale in np.arange(1.0, 0.0, -0.005):
            if w * scale < w_size and h * scale < h_size:
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

        resized.append(cv.copyMakeBorder(image, top = b_top,bottom = b_bot, left = b_left, right = b_right, borderType = cv.BORDER_CONSTANT, value = [0,0,0]))
        
    return resized
    
def get_data():
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
    
    x_train = resize_images(x_train)
    x_test = resize_images(x_test)
    
    return x_train, x_test

x_train, x_test = get_data()
x_train_new, x_test_new = normalize_data(x_train, x_test)

