
"""
Spyder Editor

This is a temporary script file.
"""

import cv2 as cv
import xml.etree.ElementTree as ET
import numpy as np
import glob
import matplotlib.pyplot as plt
import os

#def resize_bb(xmin,ymin,xmax,ymax,h,w,new_h,new_w):
    
    

def resize_image(image,h,w, max_side = 600):
    if h > w:
         new_height = max_side
         new_width = int(w * (new_height/h))
    else:
        new_width = max_side
        new_height = int(h * (new_width/w))
    resized_image = cv.resize(image, (new_width,new_height)) 
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
def save_image(image, label, image_id):

    curr_path = os.getcwd()
    path_arrabida = os.path.normpath(os.path.join(curr_path, "../dataset/resized_images/arrabida"))
    path_camara = os.path.normpath(os.path.join(curr_path, "../dataset/resized_images/camara"))
    path_clerigos = os.path.normpath(os.path.join(curr_path, "../dataset/resized_images/clerigos"))
    path_musica = os.path.normpath(os.path.join(curr_path, "../dataset/resized_images/musica"))
    path_serralves = os.path.normpath(os.path.join(curr_path, "../dataset/resized_images/serralves"))

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

    if label == 'arrabida': #arrabida
        filename = (path_arrabida + "/" + image_id + ".jpg")
        cv.imwrite(filename, image)
    elif label == 'camara': #camara
        filename = (path_camara + "/" + image_id + ".jpg")
        cv.imwrite(filename, image)
    elif label == 'clerigos': #clerigos
        filename = (path_clerigos + "/" + image_id + ".jpg")
        cv.imwrite(filename, image)
    elif label == 'musica': #musica
        filename = (path_musica + "/" + image_id + ".jpg")
        cv.imwrite(filename, image)
    elif label == 'serralves': #serralves
        filename = (path_serralves + "/" + image_id + ".jpg")
        cv.imwrite(filename, image)

#classes = ['serralves', 'musica', 'clerigos', 'camara', 'arrabida']
classes = ['musica']

for label in classes:            
    image_ids = [f.split("/")[-1].replace('.xml','') for f in glob.glob('../dataset/annotations/%s/*.xml' %(label))]
    for image_id in image_ids:
        image = cv.imread('../dataset/images/%s/%s.jpg' %(label, image_id))
        image_bb = np.copy(image)
        in_file = open('../dataset/annotations/%s/%s.xml'%(label, image_id))    
        tree=ET.parse(in_file)
        root = tree.getroot()
    
        size = root.find('size')
        h = int(size.find('height').text)
        w = int(size.find('width').text)
        
        for obj in root.iter('object'):
            xmlbox = obj.find('bndbox')    
            xmin = float(xmlbox.find('xmin').text)
            ymin = float(xmlbox.find('ymin').text)
            xmax = float(xmlbox.find('xmax').text)
            ymax = float(xmlbox.find('ymax').text)
            
         #tree = ET.parse("test.xml")
         #a = tree.find('parent')          # Get parent node from EXISTING tree
         #b = ET.SubElement(a,"child")
         #b.text = "Jay/Doctor"
         #tree.write("test.xml")
            
        cv.rectangle(image_bb,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,255,0),10)
        #cv.imshow(image_id, cv.resize(image_bb, (0,0),fx=0.2, fy=0.2) )
        #cv.waitKey(0)
        
        plt.imshow(cv.cvtColor(image_bb, cv.COLOR_BGR2RGB))
        plt.title(image_id +  " x = " + "[" + str(xmin) + ", " + str(xmax) + "]" + 
                                         " y = " + "[" + str(ymin) + ", " + str(xmax) + "]")
        plt.show()
        
        resized_image = resize_image(image,h,w)
        save_image(resized_image,label,image_id)
