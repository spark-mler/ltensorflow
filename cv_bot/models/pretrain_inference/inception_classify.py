# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Simple image classification with Inception.
Run image classification with Inception trained on ImageNet 2012 Challenge data
set.
This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.
Change the --image_file argument to any jpg image to compute a
classification of that image.
Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.
https://tensorflow.org/tutorials/image_recognition/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string("model_dir","/home/amax/tensorflow/CV_bot//script/models/inception/logs","")
tf.app.flags.DEFINE_string('model_file_name','output_graph.pb',"")
tf.app.flags.DEFINE_string('model_label_name', 'output_labels.txt',"")
tf.app.flags.DEFINE_string('image_list_file', '/home/amax/tensorflow/CV_bot/data/files_list_Testset1',
                           """Absolute path to image file.""")
tf.app.flags.DEFINE_integer('num_top_predictions', 2,
                            """Display this many predictions.""")

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long

class NodeLookup(object):
    """Converts integer node ID's to human readable labels."""

    def __init__(self,
                 uid_lookup_path=None):
        if not uid_lookup_path:
            uid_lookup_path = os.path.join(
                    FLAGS.model_dir, 'output_labels.txt')
        self.node_lookup = self.load( uid_lookup_path)

        submit_uid_lookup_path = os.path.join(
                    FLAGS.model_dir, 'submit_labels.txt')
        if not tf.gfile.Exists(submit_uid_lookup_path):
            tf.logging.fatal('File does not exist %s', submit_uid_lookup_path)
        proto_as_ascii_lines = tf.gfile.GFile(submit_uid_lookup_path).readlines()
        submit_labels = {}
        uid = 0
        for line in proto_as_ascii_lines:
            submit_labels[line.rsplit('\n')[0]] = uid
            uid += 1


        self.convert_labels = {}
        for (key, value) in self.node_lookup.items():
            self.convert_labels[key] = submit_labels[value[0]]

    def  nodeid_to_submitid(self, node_id):
        if node_id not in self.convert_labels:
            return ''
        return self.convert_labels[node_id]

    def load(self, uid_lookup_path):
        """Loads a human readable English name for each softmax node.
        Args:
          label_lookup_path: string UID to integer node ID.
          uid_lookup_path: string UID to human-readable string.
        Returns:
          dict from integer node ID to human-readable string.
        """
        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal('File does not exist %s', uid_lookup_path)


        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        total_count=0
        #p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_as_ascii_lines:
            uid_to_human[total_count] = line.rsplit('\n')
            total_count +=1
        return uid_to_human

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]


def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join(
            FLAGS.model_dir, 'output_graph.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image, sess, node_lookup):
    """Runs inference on an image.
    Args:
      image: Image file name.
    Returns:
      Nothing
    """
    if not tf.gfile.Exists(image):
        tf.logging.fatal('File does not exist %s', image)
    image_data = tf.gfile.FastGFile(image, 'rb').read()

    # Creates graph from saved GraphDef.
   

    
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)

    # Creates node ID --> English string lookup.
    

    top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
    result =  image.split("/")[8].split(".")[0] + '\t'
    for node_id in top_k:
        human_string = node_lookup.id_to_string(node_id)
        score = predictions[node_id]
        # print('%s (score = %.5f)' % (human_string, score))
        result += str(node_lookup.nodeid_to_submitid(node_id))  +'\t' + str( '%6f' % (score) ) + '\t'
    print(result)
    return result.rstrip("\t") + "\n"


def maybe_download_and_extract():
    """Download and extract model tar file."""
    dest_directory = FLAGS.model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def main(_):
    # maybe_download_and_extract()
    image_list_file = FLAGS.image_list_file
    create_graph()
    results = []
    errors  = []
    node_lookup = NodeLookup()
    i = 1
    with tf.Session() as sess:
        with open(image_list_file,'r') as fread:
            for line in fread.readlines():
                print(i)
                i += 1
                image = line.split(" ")[0]
                try:
                    results.append(run_inference_on_image(image, sess, node_lookup))
                except Exception:
                    results.append(image.split("/")[8].split(".")[0] + '\t' + '3'  +'\t' + '0.999999' + '\t' + '6'+ '\t' + '0.250145' + '\n')
                    errors.append(image)
    with open("predict/results.txt", "w") as fwrite:
        for line in enumerate(results):
            fwrite.write(line[1])
    with open('predict/errors.txt', 'w') as fwrite:
        for line in enumerate(errors):
            fwrite.write(line[1] + "\n")


if __name__ == '__main__':
    tf.app.run()
