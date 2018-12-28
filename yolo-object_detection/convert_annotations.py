#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 19:20:11 2018

@author: mariafranciscapessanha
"""

import xml.etree.ElementTree as ET
import numpy as np
import os

sets=['train', 'val', 'test']
classes = ['serralves', 'musica', 'clerigos', 'camara', 'arrabida']

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(image_set,label, image_id):
    in_file = open('../dataset/divided_sets/annotations_%s/%s/%s.xml'%(image_set,label, image_id))
    out_file = open('../dataset/divided_sets/yolo_ann_%s//%s.txt'%(image_set, image_id), 'w')
    
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

for image_set in sets:
    image_list = []
    for label in classes:
        if not os.path.exists('../dataset/divided_sets/yolo_ann_%s' %image_set):
            os.makedirs('../dataset/divided_sets/yolo_ann_%s' %(image_set))
            
            import glob
    
        image_ids = [f.split("/")[-1].replace('.xml','') for f in glob.glob('../dataset/divided_sets/annotations_%s/%s/*.xml' %(image_set,label))]
        image_list.append([f for f in glob.glob('../dataset/divided_sets/original_%s/%s/*.jpg' %(image_set,label))])
        
        for image_id in image_ids:
            convert_annotation(image_set, label, image_id)
    
    with open('../dataset/divided_sets/yolo_ann_%s/%s.txt' % (image_set,image_set), 'w') as f:
        for item in np.hstack(image_list):
            f.write("%s\n" % item)
    
    
