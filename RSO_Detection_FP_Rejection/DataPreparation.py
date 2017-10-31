

import numpy as np
import tensorflow as tf


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
        labels: the labels from the DataLabel classf.
        num_training: the number of training examples.
        num_validation: the number of cross validation examples.
        num_testing: the number of testing examples.
        store_dir: directory where the .tfrecords files will be
            stored.
    """

    def __init__(self, data_name,
                 images, labels, num_training,
                 num_validation,
                 num_testing,
                 store_dir):

        # Name of the dataset.
        self.data_name = data_name

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

            # Determine the split indicies for training.
            stop = self.num_training

            # Slice training images and labels.
            train_images = self.images[:stop]
            train_labels = self.labels[:stop]

            # Write to TFRecord.
            DataPreparation.convert_to_tfrecords(
                self.data_name + 'train.tfrecords',
                train_images,
                train_labels)

        if self.num_validation:

            # Determine the split indicies for validation.
            start = self.num_training
            stop = self.num_training + self.num_validation

            # Slice validation images and labels.
            validation_images = self.images[start:stop]
            validation_labels = self.labels[start:stop]

            # Write to TFRecord.
            DataPreparation.convert_to_tfrecords(
                self.data_name + 'validation.tfrecords',
                validation_images,
                validation_labels)

        if self.num_testing:

            # Determine the split indicies for testing.
            start = self.num_training + self.num_validation
            stop = self.num_training + self.num_validation + self.num_testing

            # Slice testing images and labels.
            test_images = self.images[start:stop]
            test_labels = self.labels[start:stop]

            # Write to TFRecord.
            DataPreparation.convert_to_tfrecords(
                self.data_name + 'test.tfrecords',
                test_images,
                test_labels)
