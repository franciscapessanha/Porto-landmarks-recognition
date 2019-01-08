# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf

import glob
import os
from sklearn.metrics import confusion_matrix

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


if __name__ == "__main__":
  model_file = "output_graph_musica.pb"
  label_file = "output_labels_musica.txt"
  input_height = 299
  input_width = 299
  input_mean = 0
  input_std = 255
  input_layer = "Placeholder"
  output_layer = "final_result"

  graph = load_graph(model_file)

  labels = load_labels(label_file)

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name)
  output_operation = graph.get_operation_by_name(output_name)

  tf.logging.set_verbosity(0)

  ground_truth = []
  prediction = []

  with open("result_musica.txt", "a") as result_file:
    result_file.truncate(0)
    with tf.Session(graph=graph) as sess:
      file_names = glob.glob('../dataset/tier3-musica/divided_sets/test/*/*.jpg')
      for file_name in file_names:
        t = read_tensor_from_image_file(
            file_name,
            input_height=input_height,
            input_width=input_width,
            input_mean=input_mean,
            input_std=input_std)

        results = np.squeeze(sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: t
        }))

        gt_label = file_name.split('/')[-2]
        output = file_name.split('/')[-1]
        top_k = results.argsort()[-5:][::-1]
        labeled = False
        for i in top_k:
          if results[i] > 0.8:
            ground_truth.append(gt_label)
            prediction.append(labels[i])
            output += " %s %s %.2f" % (gt_label, labels[i], results[i])
            labeled = True
            break

        if not labeled:
          ground_truth.append(gt_label)
          prediction.append('outro')
          output += " %s %s" % (gt_label, 'outro')

        result_file.write(output + "\n")
        print(output)

c_matrix = confusion_matrix(ground_truth, prediction, labels = ['frente', 'tras', 'lado', 'outro'] )

print(confusion_matrix)
#%%%
print(c_matrix)
FP = c_matrix.sum(axis=0) - np.diag(c_matrix)  
FN = c_matrix.sum(axis=1) - np.diag(c_matrix)
TP = np.diag(c_matrix)
TN = sum(c_matrix.ravel())- (FP + FN + TP)

accuracy = []
precision = []
recall = []

accuracy = (TP + TN)/(TP + TN + FP + FN)
precision = TP/(TP + FP)
recall = TP/(TP + FN)

acc_global = (sum(TP) + sum(TN))/(sum(TP)  + sum(TN)  + sum(FP)  + sum(FN))
precision_global = sum(TP) /(sum(TP)  + sum(FP))
recall_global = sum(TP) /(sum(TP) + sum(FN))

classes = ['frente', 'tras', 'lado', 'outro']

for i in range(len(classes)):
    print(classes[i])
    print("=================")
    print("Accuracy = %.2f \nPrecision = %.2f \nRecall = %.2f \n" % (accuracy[i],precision[i], recall[i]))

print("Global Metrics\n=================")
print("Accuracy = %.2f \nPrecision = %.2f \nRecall = %.2f \n" % (acc_global,precision_global, recall_global))
