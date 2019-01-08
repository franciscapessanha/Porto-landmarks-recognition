# Import packages
import os
import cv2 as cv
import numpy as np
import tensorflow as tf
import sys
import glob
import xml.etree.ElementTree as ET
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Grab path to current working directory
CWD_PATH = os.getcwd()

PATH_TO_MODEL = os.path.join(CWD_PATH, 'models', 'ssd_mobilenet_v1.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH, 'labels.pbtxt')

TF_SESSION, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections, category_index = None, None, None, None, None, None, None

FP, TN, FN, TP = 0, 0, 0, 0

def load_resources():
    global TF_SESSION, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections, category_index
    # Load the label
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=5, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        TF_SESSION = tf.Session(graph=detection_graph)

    # Load inputs and outputs
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

def load_image(image_name):
    path = os.path.normpath(os.path.join(CWD_PATH, image_name))
    if not os.path.isfile(path):
        return None
    image = cv.imread(path)
    return image

def resize_image(image, max_side = 600):
    h, w, _ = image.shape
    if h > w:
        new_height = max_side
        new_width = int(w * (new_height/h))
    else:
        new_width = max_side
        new_height = int(h * (new_width/w))
    return cv.resize(image, (new_width, new_height))

def label_folder():
    load_resources()
    save_path = '../resized_dataset/labeled/ssd_mobilenet'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    file_names = glob.glob('../resized_dataset/divided_sets/test/*.jpg')
    start = time.time()
    for file in file_names:
    # for i in range(3, 6):
    #     file = file_names[i]
        labeled_image = label_image(file)
        cv.imwrite(save_path + '/' + file.split("/")[-1], labeled_image)
    end = time.time()
    print('Elapsed time: %f' % (end - start))

def decode_label(index):
    return ['serralves', 'musica', 'arrabida', 'clerigos', 'camara'][int(index) - 1] if int(index) > 0 else 'control'

def decode_box(box, image):
    h, w, _ = image.shape
    return int(box[1] * w), int(box[0] * h), int(box[3] * w), int(box[2] * h)

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

def label_image(path):
    global TP, FP, TN, FN
    image = cv.imread(path)
    output = path.split("/")[-1].ljust(20)
    gt_label = output.split('-')[0]
    output += ' ' + gt_label

    # Expand image dimensions to have shape: [1, None, None, 3]
    image_expanded = np.expand_dims(image, axis=0)

    # Perform the detection
    (boxes, scores, classes, num) = TF_SESSION.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_expanded})
    pred_label = decode_label(classes[0][0])

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

        if decode_box(boxes[0][0], image) == (0, 0, 0, 0) or scores[0][0] < 0.5:
            FN += 1
            output += ' no bounding box'

        else:
            if gt_label == pred_label:
                iou = get_iou(*decode_box(boxes[0][0], image), gt_xmin, gt_ymin, gt_xmax, gt_ymax)
                if iou > 0.5:
                    TP += 1
                    output += ' %s | IoU: %.2f | %d%%' % (pred_label, iou, int(scores[0][0] * 100))
                else:
                    FP += 1
                    output += ' %s | IoU: %.2f | %d%%' % (pred_label, iou, int(scores[0][0] * 100))
            else:
                output += ' ' + pred_label
                FP += 1

    except OSError as e:
        if decode_box(boxes[0][0], image) == (0, 0, 0, 0) or scores[0][0] < 0.5:
            TN += 1
            output += ' control'
        else:
            FP += 1
            output += ' %s | %d%%' % (pred_label, int(scores[0][0] * 100))

    image = resize_image(image)
    
    # Draw the results of the detection (aka 'visualize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.5)

    print(output)
    # display_image(image)

    return image

def display_image(image):
    cv.imshow('Object detector', image)

    key = cv.waitKey(0)

    cv.destroyAllWindows()

def main():
    label_folder()

    print('\n\nTP: %d | FP: %d' % (TP, FP))
    print('TN: %d | FN: %d' % (TN, FN))

if __name__ == "__main__":
    main()