import cv2 as cv
import xml.etree.ElementTree as ET
import numpy as np
import glob
import os

class SmallBoundingBox(Exception):
    pass

def resize_image(image, h, w, max_side = 600):
    if h > w:
        new_height = max_side
        new_width = int(w * (new_height/h))
    else:
        new_width = max_side
        new_height = int(h * (new_width/w))
    resized_image = cv.resize(image, (new_width, new_height))
    return resized_image

def create_directories(file_type, label):
    path = os.path.normpath(os.path.join(os.getcwd(), "../resized_dataset/%s/%s" % (file_type, label)))

    if not os.path.exists(path):
        os.makedirs(path)

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
    filename = (os.path.normpath(os.path.join(os.getcwd(), "../resized_dataset/images/%s" % label)) + "/" + image_id + ".jpg")
    cv.imwrite(filename, image)

def validate_and_recize_bounding_box(h, w, xmin, ymin, xmax, ymax, max_side = 600):
    if h > w:
        new_height = max_side
        new_width = int(w * (new_height/h))
    else:
        new_width = max_side
        new_height = int(h * (new_width/w))

    xmin = 0 if xmin < 0 else int((xmin * new_width)/w)
    ymin = 0 if ymin < 0 else int((ymin * new_height)/h)
    xmax = new_width if xmax > w else int((xmax * new_width)/w)
    ymax = new_height if ymax > h else int((ymax * new_height)/h)

    if xmax - xmin < 32 or ymax - ymin < 32:
        return None, None, None, None, None, None

    return new_height, new_width, xmin, ymin, xmax, ymax

classes = ['serralves', 'musica', 'clerigos', 'camara', 'arrabida']

for label in classes:
    create_directories('images', label)
    create_directories('annotations', label)

    image_ids = [f.split("/")[-1].replace('.xml','') for f in glob.glob('../dataset/annotations/%s/*.xml' %(label))]
    for image_id in image_ids:
        image = cv.imread('../dataset/images/%s/%s.jpg' %(label, image_id))
        image_bb = np.copy(image)
        xml_annotation = open('../dataset/annotations/%s/%s.xml'%(label, image_id))
        tree = ET.parse(xml_annotation)
        root = tree.getroot()

        size = root.find('size')
        h = int(size.find('height').text)
        w = int(size.find('width').text)

        try:
            for obj in root.iter('object'):
                xmlbox = obj.find('bndbox')
                xmin = float(xmlbox.find('xmin').text)
                ymin = float(xmlbox.find('ymin').text)
                xmax = float(xmlbox.find('xmax').text)
                ymax = float(xmlbox.find('ymax').text)

                new_h, new_w, new_xmin, new_ymin, new_xmax, new_ymax = validate_and_recize_bounding_box(h, w, xmin, ymin, xmax, ymax)
                if new_h is None:
                    raise SmallBoundingBox

                xmlbox.find('xmin').text = str(new_xmin)
                xmlbox.find('ymin').text = str(new_ymin)
                xmlbox.find('xmax').text = str(new_xmax)
                xmlbox.find('ymax').text = str(new_ymax)
        except SmallBoundingBox:
            print('%s | Bounding box to small' % image_id)
            continue

        size.find('height').text = str(new_h)
        size.find('width').text = str(new_w)

        print("%s | [%d, %d] | x = [%d, %d] y = [%d, %d]" % (image_id, new_w, new_h, new_xmin, new_xmax, new_ymin, new_ymax))
        resized_image = resize_image(image, h, w)

        tree.write(open('../resized_dataset/annotations/%s/%s.xml'%(label, image_id), 'w'), encoding='unicode')

        # cv.rectangle(resized_image,(new_xmin, new_ymin), (new_xmax, new_ymax), (0, 255, 0), 2)
        # cv.imshow(image_id, resized_image)
        # cv.waitKey(0)
        # cv.destroyWindow(image_id)

        save_image(resize_image(image, h, w), label, image_id)
