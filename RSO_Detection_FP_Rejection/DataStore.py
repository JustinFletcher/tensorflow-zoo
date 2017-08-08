

import csv
from SpaceShotOutput import SpaceShotOutput
from AstroGraphOutput import AstroGraphOutput
import h5py
import numpy as np


def mean(list_of_strfloats):

    """
    Helper function to compute the mean of a list of floating
    point numbers that are strings.

    Input:
        list_of_strfloats: list of floating point numbers (as
            strings).

    Output:
        avg: mean of the numbers in the list.
    """

    summation = float(0)

    for i in range(len(list_of_strfloats)):

        summation += float(list_of_strfloats[i])

    avg = summation / len(list_of_strfloats)

    return avg


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


class DataStore(object):

    """
    Class to sort and store image stacks with the corresponding
    human labeled information from the .csv. The data is stored
    in a .hdf5 file.
    
    Attributes:
        csv_file_name: name of the .csv file where the human
            labeled information is stored.
        fits_file_dir: directory where the .fits files are
            stored.
        ast_file_dir: directory where the .ast files are
            stored.
        hdf5_file_name: name of the output .hdf5 file where
            the sorted data is stored.
    """

    def __init__(self, csv_file_name, fits_file_dir,
                 ast_file_dir, hdf5_file_name,
                 hdf5_file_dir):

        # Name of the .csv file.
        self.csv_file_name = csv_file_name

        # Path to where the .fits files are stored.
        self.fits_file_dir = fix_path(fits_file_dir)

        # Path to where the .ast files are stored.
        self.ast_file_dir = fix_path(ast_file_dir)

        # Path to where the .hdf5 file will be stored.
        self.hdf5_file_dir = fix_path(hdf5_file_dir)

        # Name of the .hdf5 file.
        self.hdf5_file_name = hdf5_file_name

        # List of the names of the individual datasets
        # in the .hdf5. These names will be required
        # when reading in frame stacks for labeling.
        self.dataset_names = []

        # Call to write the data to the hdf5 file.
        self.write_hdf5()

    def write_hdf5(self):

        # Open the .csv file.
        with open(self.csv_file_name, 'r') as f:

            # Read the rows of the .csv, discarding the first
            # row that contains the column names.
            reader = csv.reader(f)
            rows = [row for row in reader]
            rows = rows[1:]

            # Initialize variables.
            ast_file = []
            xcenter = []
            ycenter = []
            human = []
            astrograph = []
            object_name = []
            image_number = []
            group_number = []
            track_number = []
            fits_file = []
            file_number = []

            # Append the information from each row to the
            # corresponding variable.
            for row in rows:

                ast_file.append(row[0])
                xcenter.append(row[1])
                ycenter.append(row[2])
                human.append(row[3])
                astrograph.append(row[4])
                object_name.append(row[5])
                image_number.append(row[7])
                group_number.append(row[8])
                track_number.append(row[11])
                fits_file.append(row[12])
                file_number.append(row[13])

            # Find the unique track numbers (since they are
            # repeated in the .csv file).
            unique_track_numbers = list(set(track_number))
            self.dataset_names = unique_track_numbers

            # The file where the sorted data is stored.
            sorted_datasets_f = h5py.File(self.hdf5_file_dir
                                          + self.hdf5_file_name,
                                          'a')

            # Loop over track numbers (frame stacks) sorting the
            # human labeled information with each image stack.
            for track_num in unique_track_numbers:

                # Find the indices of the current track number.
                track_number_indices = [num for num in
                                        range(len(track_number))
                                        if track_number[num]
                                        == track_num]

                # Get the .ast file names.
                ast_files = [ast_file[num] for num
                             in track_number_indices]

                # Get the .fits file names.
                fits_files = [fits_file[num] for num
                              in track_number_indices]

                # Get the detected object names.
                object_names = [object_name[num] for num
                                in track_number_indices]

                # Get the x-locations of the detected objects'
                # centroids.
                xcenters = [xcenter[num] for num
                            in track_number_indices]

                # Get the y-locations of the detected objects'
                # centroids.
                ycenters = [ycenter[num] for num
                            in track_number_indices]

                # Human decision labels.
                human_labels = [human[num] for num
                                in track_number_indices]

                # AstroGraph decision labels.
                astrograph_labels = [astrograph[num] for num
                                     in track_number_indices]

                # Create a list of the unique .ast file names.
                # A simpler, one-liner, for doing this is
                # unique_ast_files = list(set(ast_files)).
                # However, it appears that file order is not
                # always preserved when using this approach.
                # The for loop below is used to preserve the
                # file order.
                unique_ast_files = []
                for file in ast_files:

                    if file not in unique_ast_files:

                        unique_ast_files.append(file)

                # Create a list of the unique .fits file names.
                # A simpler, one-liner, for doing this is
                # unique_fits_files = list(set(fits_files)).
                # However, it appears that file order is not
                # always preserved when using this approach.
                # The for loop below is used to preserve the
                # file order.
                unique_fits_files = []
                for file in fits_files:

                    if file not in unique_fits_files:

                        unique_fits_files.append(file)

                # Get the frame numbers from the .ast files for
                # sorting.
                AGO = AstroGraphOutput(self.ast_file_dir,
                                       unique_ast_files)
                frame_numbers = AGO.frame_number
                total_frame_numbers = AGO.total_frames

                # TODO: sorting algorithm to ensure frames
                # are in order. Sorting may not be necessary if
                # the frames in the .csv file are in order.

                # Create a list of the unique object names.
                unique_object_names = list(set(object_names))

                # Initialize variables for recording.
                object_xcenter = []
                object_ycenter = []
                mean_object_centroid = []
                label = []

                # Loop to record the mean object centroid and
                # detection labels for detections in a frame stack.
                for j in range(len(unique_object_names)):

                    object_xcenter.append([])
                    object_ycenter.append([])

                    # mean_object_centroid is a list of lists of
                    # detected object centroids in the format [x, y].
                    # The length of the list is equal to the number
                    # of detections in a given frame stack. An
                    # example mean_object_centroid is:
                    # [[235.4, 116.8], [210.2, 99.8], [136.8, 54.2]]
                    # for 3 objects detected in a frame stack.
                    mean_object_centroid.append([None, None])

                    # label is a list of lists of human and
                    # AstroGraph detection classifications. The
                    # length of the list is equal to the number of
                    # detections in a given frame stack. An example
                    # label is:
                    # [[human_1, AG_1], [human_2, AG_2], [human_3, AG_3]]
                    # for 3 objects detected in a frame stack.
                    label.append([None, None])
                    name_ind = [name for name
                                in range(len(object_names))
                                if object_names[name]
                                == unique_object_names[j]]

                    for k in range(len(name_ind)):

                        if k == 0:

                            if human_labels[name_ind[k]] != 'NaN':

                                label[j][0] = \
                                    int(human_labels[name_ind[k]])
                                label[j][1] = \
                                    int(astrograph_labels[name_ind[k]])

                        object_xcenter[j].append(xcenters[name_ind[k]])
                        object_ycenter[j].append(ycenters[name_ind[k]])

                    mean_object_centroid[j][0] = mean(
                        object_xcenter[j])
                    mean_object_centroid[j][1] = mean(
                        object_ycenter[j])

                SSO = SpaceShotOutput(self.fits_file_dir,
                                      unique_fits_files)

                frame_stack = sorted_datasets_f.create_dataset(
                    track_num,
                    (len(SSO.images),
                     len(SSO.images[0]),
                     len(SSO.images[0][0])),
                    compression="gzip",
                    compression_opts=4)

                frame_stack[:, :, :] = SSO.images
                frame_stack.attrs['label'] = np.asarray(label, 'uint8')
                frame_stack.attrs['centroid'] = mean_object_centroid

            sorted_datasets_f.close()