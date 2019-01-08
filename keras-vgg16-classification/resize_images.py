import os
import numpy as np
import cv2 as cv
from keras.preprocessing.image import img_to_array, load_img
import shutil
import glob
from keras.applications import vgg16

sets = ['train', 'val', 'test']
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
    """
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
    """
    #image = load_img(image_id, target_size = (224,224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    resized_image = vgg16.preprocess_input(image)
    return np.squeeze(resized_image)

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
    path_folder = os.path.normpath(os.path.join(curr_path, "../dataset/vgg16_resized_sets", folder_name, label))

    if not os.path.exists(path_folder):
        os.makedirs(path_folder)
   
    filename = path_folder + path[path.rindex('/'):]
    cv.imwrite(filename, image)
    
"""
Run
==============================================================================
"""
def run():
    curr_path = os.getcwd()
    resized_sets_folder = os.path.normpath(os.path.join(curr_path, "../dataset/vgg16_resized_sets"))
    if os.path.exists(resized_sets_folder):
        shutil.rmtree(resized_sets_folder)
        
    for image_set in sets:
        print('%s' %image_set)
        curr_path = os.getcwd()
        set_path = os.path.normpath(os.path.join(curr_path, "../dataset/divided_sets/", image_set))
        classes = os.listdir(set_path)
            
        for label in classes: 
            image_ids = [f for f in glob.glob('../dataset/divided_sets/%s/%s/*.jpg' % (image_set, label))]
            
            for image_id in image_ids:
                image = cv.imread(image_id)
                image = resize_images(image)
                save_image(image_id, image, label, image_set)
               
       
run()