

import h5py
from Chipper import Chipper
import numpy as np


def fix_path(path):

    """
    Helper function to ensure the path has the proper structure.

    Input:
        path: path to a directory.

    Output:
        path: path to a directory with the proper structure
            enforced.
    """

    if path[-1] != '/':

        path += '/'

    return path


class DataLabel(object):

    """
    Class to label the chipped frame stacks with a one-hot
    encoding scheme.

    Attributes:
        hdf5_file_dir: directory where the .hdf5 file is
            stored.
        hdf5_file_name: name of the .hdf5 file where
            the sorted data is stored.
        dataset_names: names of the datasets in the .hdf5
            file. These names are generated in the
            DataStore class and are an attribute of that
            class.
        images: the chipped frame stacks that will be used
            for training/cross validation/testing.
        labels: labels for each image. The labels are
            one-hot encoded and contain 4 classes.
    """

    def __init__(self, hdf5_file_dir, hdf5_file_name,
                 dataset_names, chip_width,
                 chip_height):

        # Path to where the .fits files are stored.
        self.hdf5_file_dir = fix_path(hdf5_file_dir)

        # Name of the .hdf5 file.
        self.hdf5_file_name = hdf5_file_name

        # Names of the datasets in the .hdf5 file.
        self.dataset_names = dataset_names

        # Chip width and height.
        self.chip_width = chip_width
        self.chip_height = chip_height

        # Frame stacks.
        self.images = []

        # Labels (using one-hot encoding). The structure
        # of the labels is: [TP, FP, TN, FN]. For example,
        # a FP would be labeled as [0, 1, 0, 0].
        self.labels = []

        # Call to chip the frame stacks and label the
        # chip stacks.
        self.chip_and_label_datasets()

    def chip_and_label_datasets(self):

        # Read the .hdf5 file.
        sorted_datasets_f = h5py.File(self.hdf5_file_dir
                                      + self.hdf5_file_name,
                                      'a')

        # Loop over each collect.
        for i in range(len(sorted_datasets_f)):

            # Frames in each collect.
            frame_stack = sorted_datasets_f[
                self.dataset_names[i]]

            # Corresponding human and AstroGraph labels in
            # the collect.
            label = frame_stack.attrs['label']

            # Corresponding mean RSO centroids in the
            # collect.
            mean_object_centroid = frame_stack.attrs[
                'centroid']

            # Loop over the detections in the collect.
            for j in range(len(label)):

                # Chip the frame stack around the mean
                # RSO centroid.
                chipper = Chipper(
                    frame_stack,
                    mean_object_centroid[j],
                    self.chip_width,
                    self.chip_height)

                # Add the cip stack to the collection of
                # chip stacks.
                self.images.append(chipper.chip_stack)

                # Label the chip stack using one-hot
                # encoding.
                if label[j][1] and label[j][0]:

                    # TP
                    self.labels.append(
                        np.array([1, 0, 0, 0], 'uint8'))

                elif label[j][1] and not label[j][0]:

                    # FP
                    self.labels.append(
                        np.array([0, 1, 0, 0], 'uint8'))

                elif not label[j][1] and not label[j][0]:

                    # TN
                    self.labels.append(
                        np.array([0, 0, 1, 0], 'uint8'))

                elif not label[j][1] and label[j][0]:

                    # FN
                    self.labels.append(
                        np.array([0, 0, 0, 1], 'uint8'))

                elif label[j][0] is None:

                    self.labels.append(None)

                else:

                    print('Something is wrong with the labels'
                          + ' in the .hdf5 file. Please fix before'
                          + ' continuing')

                    break

