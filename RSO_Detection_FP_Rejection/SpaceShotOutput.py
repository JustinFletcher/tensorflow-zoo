

import os
from AstronomyImage import AstronomyImage


def fix_path(path):

    """
    Helper function to ensure the path has the proper
    structure.

    Input:
        path: path to a directory.

    Output:
        path: path to a directory with the proper
        structure enforced.
    """

    if path[-1] != '/':

        path += '/'

    return path


class SpaceShotOutput(object):

    """
    Class to parse either all of the .fits files in
    a directory or a list of .fits files provided
    by the user.

    Attributes:
        dir_path: path to directory where files are
            stored
        file_list: list of .fits files. Only required
            if parsing a specified list of .fits files
        images: the images in the .fits files
        headers: the headers in the .fits files
    """

    def __init__(self, dir_path, file_list=None):

        # Path to directory where the .fits files
        #  are stored.
        self.dir_path = fix_path(dir_path)

        # List of .fits files. Only required if
        #  parsing a defined list of .fits files.
        self.file_list = file_list

        # Initialize the list of images.
        self.images = []

        # Initialize the list of headers.
        self.headers = []

        # Call to extract the images and headers
        # from the stack of .fits files.
        self.get_images()

    def get_images(self):

        if not self.file_list:

            # Initialize the list of .fits files.
            files = []

            # Avoid placing hidden directories in
            #  the list of files.
            for file in os.listdir(self.dir_path):

                if not file.startswith('.'):

                    files.append(file)

            # Sort the files.
            files = sorted(files)

            # Pull the files from the directory
            #  that have the .ast extension.
            for file in files:

                # Separate the file names from
                #  the extensions.
                file_name, file_extension =\
                    os.path.splitext(file)

                # Parse the current .fits file.
                if file_extension == '.fits':

                    # Parse the .fits file.
                    astronomy_image = AstronomyImage(
                        self.dir_path + file)

                    # Add the image to the collection.
                    self.images.append(
                        astronomy_image.image)

                    # Add the header to the collection.
                    self.headers.append(
                        astronomy_image.header)

        if self.file_list:

            for file in self.file_list:

                # Parse the .fits file.
                astronomy_image = AstronomyImage(
                    self.dir_path + file)

                # Add the image to the collection.
                self.images.append(
                    astronomy_image.image)

                # Add the header to the collection.
                self.headers.append(
                    astronomy_image.header)