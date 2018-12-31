
"""
Spyder Editor

This is a temporary script file.
"""

import cv2 as cv
import xml.etree.ElementTree as ET
import numpy as np
import glob
import matplotlib.pyplot as plt

classes = ['serralves', 'musica', 'clerigos', 'camara', 'arrabida']

for label in classes:            
    image_ids = [f.split("/")[-1].replace('.xml','') for f in glob.glob('../dataset/annotations/%s/*.xml' %(label))]
    for image_id in image_ids:
        image_bb = np.copy(cv.imread('../dataset/images/%s/%s.jpg' %(label, image_id)))
        in_file = open('../dataset/annotations/%s/%s.xml'%(label, image_id))    
        tree=ET.parse(in_file)
        root = tree.getroot()
    
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            
            
            xmlbox = obj.find('bndbox')    
            xmin = float(xmlbox.find('xmin').text)
            ymin = float(xmlbox.find('ymin').text)
            xmax = float(xmlbox.find('xmax').text)
            ymax = float(xmlbox.find('ymax').text)
            
        cv.rectangle(image_bb,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,255,0),2)
        plt.imshow(cv.cvtColor(image_bb, cv.COLOR_BGR2RGB))
        plt.title(image_id +  " x = " + "[" + str(xmin) + ", " + str(xmax) + "]" + 
                                       " y = " + "[" + str(ymin) + ", " + str(xmax) + "]")
        plt.show()

        print("Shape image = ", np.shape(image_bb))
        
    
