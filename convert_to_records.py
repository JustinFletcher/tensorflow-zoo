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

"""Converts MNIST data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import mnist

FLAGS = None


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfrecord(stimuli, targets, num_examples, name):
    """Converts a dataset to tfrecords."""

    # Check for data/label count consistency.
    if stimuli.shape[0] != num_examples:
        raise ValueError('Images size %d does not match label size %d.' %
                         (stimuli.shape[0], num_examples))

    # Extract shape information to local variables.
    rows = stimuli.shape[1]
    cols = stimuli.shape[2]
    depth = stimuli.shape[3]

    # Build a writer for the tfrecord.
    filename = os.path.join(FLAGS.directory, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)

    # Iterate over the examples.
    for index in range(num_examples):
        image_raw = stimuli[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(targets[index])),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()


def main(unused_argv):

    # Get the data.
    data_sets = mnist.read_data_sets(FLAGS.directory,
                                     dtype=tf.uint8,
                                     reshape=False,
                                     validation_size=FLAGS.validation_size)
    # Read Iris data here.

    # Extract train/test splits here.
    # Convert to Examples and write the result to TFRecords.
    convert_to_tfrecord(data_sets.train.images,
                        data_sets.train.labels,
                        data_sets.train.num_examples, 'train')
    convert_to_tfrecord(data_sets.validation.images,
                        data_sets.validation.labels,
                        data_sets.validation.num_examples, 'validation')
    convert_to_tfrecord(data_sets.test.images,
                        data_sets.test.labels,
                        data_sets.test.num_examples, 'test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--directory',
        type=str,
        default='../data',
        help='Directory to download data files and write the converted result'
    )
    parser.add_argument(
        '--validation_size',
        type=int,
        default=5000,
        help="""\
        Number of examples to separate from the training data for validation.\
        """
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
