# Import packages
import os
import cv2 as cv
import numpy as np
import tensorflow as tf
import sys

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Grab path to current working directory
CWD_PATH = os.getcwd()

PATH_TO_MODEL = os.path.join(CWD_PATH, 'models', 'ssd_mobilenet_v1.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH, 'labels.pbtxt')

TF_SESSION, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections, category_index = None, None, None, None, None, None, None

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

def label_image(image):
    # Expand image dimensions to have shape: [1, None, None, 3]
    image_expanded = np.expand_dims(image, axis=0)

    # Perform the detection
    (boxes, scores, classes, num) = TF_SESSION.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_expanded})

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
        min_score_thresh=0.8)

    return image

def display_image(image, argv_input):
    # All the results have been drawn on image. Now display the image.
    cv.imshow('Object detector', image)
    print('Press \'s\' to save the labeled image.\nPress \'Esc\' to close.')

    # Press any key to close the image
    while(True):
        key = cv.waitKey(0)
        if key == 115:
            name, extension = argv_input.split('/')[-1].split('.')
            cv.imwrite('%s-labeled.%s' % (name, extension), image)
            break
        elif key == 27:
            break

    cv.destroyAllWindows()



def print_usage():
    print('Porto Monuments Image Recognition v0.1\n' +
            '   usage: porto-image-recognition [image]\n')

def main():
    if len(sys.argv) > 2:
        print('To many arguments passed.')
        print_usage()
        return
    elif len(sys.argv) == 2:
        image = load_image(sys.argv[1])
        if image is None:
            print('Image path does not exist.')
            return
        load_resources()
        labeled_image = label_image(image)
        display_image(labeled_image, sys.argv[1])



if __name__ == "__main__":
    main()