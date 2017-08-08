

import tensorflow as tf
import numpy as np


def _int64_feature(value):

    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):

    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value]))


class DataPreparation(object):

    """
    Class to convert the images and labels to tfrecords for model
    training, cross validation, and testing. The .tfrecords files
    are written to a specified directory.
    
    Attributes:
        images: the images from the DataLabel class.
        labels: the labels from the DataLabel class.
        num_training: the number of training examples.
        num_validation: the number of cross validation examples.
        num_testing: the number of testing examples.
        store_dir: directory where the .tfrecords files will be
            stored.
    """

    def __init__(self, images, labels, num_training,
                 num_validation,
                 num_testing,
                 store_dir):

        # Cropped frame stacks from the DataLabel class.
        self.images = images

        # Corresponding labels from the DataLabel class.
        self.labels = labels

        # Number of images for training.
        self.num_training = num_training

        # Number of images for validation.
        self.num_validation = num_validation

        # Number of images for testing.
        self.num_testing = num_testing

        # Directory for storing the .tfrecords files.
        self.store_dir = store_dir

        # Call to convert the datasets to .tfrecords.
        self.conversion()

    @staticmethod
    def convert_to_tfrecords(file_name, images, labels):

        """
        Converts a dataset to tfrecords.
        """

        # Get dimensions of the chip stacks.
        rows = len(images[0][0])
        cols = len(images[0][0][0])
        depth = len(images[0])

        # Initialize the .tfrecord writer.
        print('Writing', file_name)
        writer = tf.python_io.TFRecordWriter(file_name)

        # Map labels from one-hot encoding to numeric classes for
        # use with tf.nn.sparse_softmax_cross_entropy_with_logits.
        for index in range(len(labels)):

            image = np.asarray(images[index])
            image_raw = image.tostring()

            if labels[index][0] == 1:

                label = np.uint8(0)

            elif labels[index][1] == 1:

                label = np.uint8(1)

            elif labels[index][2] == 1:

                label = np.uint8(2)

            elif labels[index][3] == 1:

                label = np.uint8(3)

            else:

                print('There is an issue with your labels.')

                break

            example = tf.train.Example(
                features=tf.train.Features(feature={
                    'height': _int64_feature(rows),
                    'width': _int64_feature(cols),
                    'depth': _int64_feature(depth),
                    'label': _int64_feature(label),
                    'image_raw': _bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())

        writer.close()

    def conversion(self):

        """
        Calls the convert_to_tfrecords function 3 times: once
        for each of the .tfrecords files to be created.
        """

        # Convert to Examples and write the result to TFRecords.
        if self.num_training:

            stop = self.num_training
            train_images = self.images[:stop]
            train_labels = self.labels[:stop]
            DataPreparation.convert_to_tfrecords(
                'train.tfrecords',
                train_images,
                train_labels)

        if self.num_validation:

            start = self.num_training
            stop = self.num_training + self.num_validation
            validation_images = self.images[start:stop]
            validation_labels = self.labels[start:stop]
            DataPreparation.convert_to_tfrecords(
                'validation.tfrecords',
                validation_images,
                validation_labels)

        if self.num_testing:

            start = self.num_training + self.num_validation
            stop = self.num_training\
                   + self.num_validation\
                   + self.num_testing
            test_images = self.images[start:stop]
            test_labels = self.labels[start:stop]
            DataPreparation.convert_to_tfrecords(
                'test.tfrecords',
                test_images,
                test_labels)