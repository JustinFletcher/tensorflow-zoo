

from DataStore import DataStore
from DataLabel import DataLabel
from DataPreparation import DataPreparation
from ObjectAstrometry import ObjectAstrometry
from AstroGraphOutput import AstroGraphOutput
from AstroGraph import AstroGraph
from SeparateCollects import SeparateCollects
from AstronomyImage import AstronomyImage
from SpaceShotOutput import SpaceShotOutput
import csv
import matplotlib.pyplot as plt


# Example AstroGraph commands generated using the
# AstroGraph wrapper.
AG = AstroGraph(starDetect_threshold=2.4,
                AstroGraph_targetDetect_threshold=4.1)
print('Command to run an AstroGraph properties file: '
      + AG.run_properties_file())
print('Command to list all AstroGraph properties: '
      + AG.list_properties())
print('Command to list all AstroGraph commands: '
      + AG.list_commands())
print('Example command to perform a specific detection task: '
      + AG.generate_commandline(
    ['star detection threshold', 'target detection threshold']))


# Example parsing a single .ast file.
path_to_single_ast_file =\
    '/home/jermws/Sample Raven Images/AstroGraph/metrics/AST/sat_28884.0118.ast'
OA = ObjectAstrometry(path_to_single_ast_file)
print('Object centroid(s) in a sigle .ast file: ',
      OA.object_centroid)
print('Frame number for a single .ast file: ',
      OA.frame_number)


# Example parsing a single .fits file.
path_to_single_fits_file = '/home/jermws/Sample Raven Images/FITS/sat_28884.0001.fits'
AI = AstronomyImage(path_to_single_fits_file)
# Plot the image.
plt.imshow(AI.image, cmap=plt.get_cmap('gray'))
plt.colorbar()
plt.show()


# Example parsing a directory of .ast files.
path_to_directory_of_ast_files =\
    '/home/jermws/Sample Raven Images/AstroGraph/metrics/AST'
AGO = AstroGraphOutput(path_to_directory_of_ast_files)
print('All object centroids: ',
      AGO.object_centroid)
print('All frame numbers: ',
      AGO.frame_number)


# Example parsing a list of .ast files.
AGO_list = AstroGraphOutput(
    path_to_directory_of_ast_files,
    AGO.file_name[0:12])
print('Centroid(s) for two collects: ',
      AGO_list.object_centroid)
print('Frame numbers for two collects: ',
      AGO_list.frame_number)


# Example parsing a directory of .fits files.
path_to_directory_of_fits_files =\
    '/home/jermws/Sample Raven Images/FITS'
SSO = SpaceShotOutput(path_to_directory_of_fits_files)
# Plot the first image.
plt.imshow(SSO.images[0], cmap=plt.get_cmap('gray'))
plt.colorbar()
plt.show()


# Example separating the collects in a directory of
# .ast files.
SC = SeparateCollects(path_to_directory_of_ast_files)
print('Separated collect files: ',
      SC.collect_files)


# Example separating the collects in a list of
# .ast files.
SC_list = SeparateCollects(path_to_directory_of_ast_files,
                           AGO.file_name[0:12])
print('Separated collect files for two collects: ',
      SC_list.collect_files)


# Example storing frame stacks, labels, and mean object
# centroids in a .hdf5 file. DataStore compresses the
# .hdf5 file to reduce the required disk space.
name_of_labels_csv =\
    'dataset_labels.csv'
path_to_directory_of_fits_files =\
    '/home/jermws/Sample Raven Images/FITS'
name_of_hdf5_file_for_storage =\
    'labeled_datasets.hdf5'
path_to_directory_for_storage =\
    '/home/jermws/PycharmProjects/Machine_Learning_Updated/'
DS = DataStore(name_of_labels_csv,
               path_to_directory_of_fits_files,
               path_to_directory_of_ast_files,
               name_of_hdf5_file_for_storage,
               path_to_directory_for_storage)


# Save the list of data set names in the .hdf5 file
# so they can be recalled when it comes time to read
# the data back in.
with open('dataset_names.csv', 'w') as f:

    w = csv.writer(f)

    for row in DS.dataset_names:

        w.writerow([row])


# Read in the data set names so they can be used to
# label the chip stacks.
dataset_names = []
for row in open('dataset_names.csv'):

    dataset_names.append(row.rstrip('\n'))


# Example usage of DataLabel class to create and label
# the individual chip stacks.
DL = DataLabel(path_to_directory_for_storage,
               name_of_hdf5_file_for_storage,
               dataset_names, 50, 50)
# Plot a single chip from the first chip stack.
plt.imshow(DL.images[0][0], cmap=plt.get_cmap('gray'))
plt.colorbar()
plt.show()


# Example usage of the DataPreparation class to convert
# the labeled chip stacks to .tfrecords.
number_training = 80
number_validation = 5
number_testing = 0
DP = DataPreparation(DL.images, DL.labels,
                     number_training,
                     number_validation,
                     number_testing,
                     path_to_directory_for_storage)