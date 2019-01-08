import sys
from yolo import YOLO
from PIL import Image
import cv2 as cv
import numpy as np
import xml.etree.ElementTree as ET
import os
import glob
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

FP, TN, FN, TP = 0, 0, 0, 0

def display_image(image):
    cv.imshow('Object detector', image)

    key = cv.waitKey(0)

    cv.destroyAllWindows()

def label_folder(yolo):
    start = time.time()
    save_path = '../resized_dataset/labeled/yolo'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    file_names = glob.glob('../resized_dataset/divided_sets/test/*.jpg')
    for file in file_names:
    # for i in range(3, 6):
    #     file = file_names[i]
        labeled_image = label_image(file, yolo)
        cv.imwrite(save_path + '/' + file.split("/")[-1], labeled_image)
    end = time.time()
    print('Elapsed time: %f' % (end - start))

def get_iou(pred_xmin, pred_ymin, pred_xmax, pred_ymax, gt_xmin, gt_ymin, gt_xmax, gt_ymax):
    assert pred_xmin < pred_xmax
    assert pred_ymin < pred_ymax
    assert gt_xmin < gt_xmax
    assert gt_ymin < gt_ymax

    # determine the coordinates of the intersection rectangle
    x_left = max(pred_xmin, gt_xmin)
    y_top = max(pred_ymin, gt_ymin)
    x_right = min(pred_xmax, gt_xmax)
    y_bottom = min(pred_ymax, gt_ymax)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    pred_area = (pred_xmax - pred_xmin) * (pred_ymax - pred_ymin)
    gt_area = (gt_xmax - gt_xmin) * (gt_ymax - gt_ymin)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(pred_area + gt_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def label_image(path, yolo):
    global TP, FP, TN, FN
    output = path.split("/")[-1].ljust(20)
    gt_label = output.split('-')[0]
    output += ' ' + gt_label

    # Perform the detection
    image, info, pred_xmin, pred_ymin, pred_xmax, pred_ymax = yolo.detect_image(Image.open(path))
    image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
    pred_label = info.split(' ')[0] if info is not None else None
    score = float(info.split(' ')[1]) if info is not None else None

    try:
        xml_annotation = open(path.replace('.jpg', '.xml'))
        tree = ET.parse(xml_annotation)
        root = tree.getroot()

        for obj in root.iter('object'):
            xmlbox = obj.find('bndbox')
            gt_xmin = int(xmlbox.find('xmin').text)
            gt_ymin = int(xmlbox.find('ymin').text)
            gt_xmax = int(xmlbox.find('xmax').text)
            gt_ymax = int(xmlbox.find('ymax').text)

        if pred_xmin is None or score < 0.5:
            FN += 1
            output += ' no bounding box'

        else:
            if gt_label == pred_label:
                iou = get_iou(pred_xmin, pred_ymin, pred_xmax, pred_ymax, gt_xmin, gt_ymin, gt_xmax, gt_ymax)
                if iou > 0.5:
                    TP += 1
                    output += ' %s | IoU: %.2f | %d%%' % (pred_label, iou, int(score * 100))
                else:
                    FP += 1
                    output += ' %s | IoU: %.2f | %d%%' % (pred_label, iou, int(score * 100))
            else:
                output += ' ' + pred_label
                FP += 1

    except OSError as e:
        if pred_xmin is None or score < 0.5:
            TN += 1
            output += ' control'
        else:
            FP += 1
            output += ' %s | %d%%' % (pred_label, int(score * 100))

    print(output)
    # display_image(image)

    return image

if __name__ == '__main__':
    yolo = YOLO(**{
      'image': True,
      'output': './test_data/detection_result_images',
      'model_path': 'output/model.h5',
      'classes_path': 'porto_dataset/porto_classes.txt'
    })

    label_folder(yolo)

    yolo.close_session()

    print('\nTP: %d | FP: %d' % (TP, FP))
    print('TN: %d | FN: %d' % (TN, FN))

