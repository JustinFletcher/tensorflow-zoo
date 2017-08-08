

from AstroGraphOutput import AstroGraphOutput


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


class SeparateCollects(object):
    
    """
    Class to separate a list of .ast files into groups
    of .ast files corresponding to individual collects.
    This class assumes the files are in order. If they
    aren't in order, then an additional script will
    need to be added to sort the files in order first
    based on the frame numbers. This class is optional
    in the overall data flow pipeline.

    Attributes:
        dir_path: path to directory where files are
            stored
        fileList: list of .ast files. Only required
            if parsing a specified list of .ast files
        collect_files: list of .ast files in each
            collect
    """

    def __init__(self, dirPath, fileList=None):

        # List of .ast files in each collect.
        self.collect_files = []

        # Path to directory where the .ast files are
        # stored.
        self.dirPath = fix_path(dirPath)

        # List of .ast files. Only required if parsing
        # a defined list of .ast files.
        self.fileList = fileList

        # Call to separate the .ast file names into
        # individual collects.
        self.separate_collects()

    def separate_collects(self):

        if not self.fileList:

            # Initialize files processed.
            files_processed = []

            # Create an AstroGraphOutput object for all
            # the files in the directory.
            astrograph_output = AstroGraphOutput(self.dirPath)

            # Loop to separate the individual collects.
            for i in range(len(
                    astrograph_output.frame_number)):

                # Keep track of the files that have been
                # processed.
                files_processed.append(
                    astrograph_output.file_name[i])

                # Continue through conditional once last
                # frame in a collect has been reached.
                if astrograph_output.frame_number[i]\
                        % astrograph_output.total_frames[i]\
                        == 0:

                    # Add the set of files in a single
                    # collect to the list of collects in this
                    # directory.
                    self.collect_files.append(files_processed)

                    # Reset the files that have been processed.
                    files_processed = []

        if self.fileList:

            # Initialize files processed.
            files_processed = []

            # Create an AstroGraphOutput object for all
            # the files in the file list.
            astrograph_output = AstroGraphOutput(self.dirPath,
                                                 self.fileList)

            # Loop to separate the individual collects.
            for i in range(len(self.fileList)):

                # Keep track of the files that have been processed.
                files_processed.append(
                    astrograph_output.file_name[i])

                # Continue through conditional once last frame in
                # a collect has been reached.
                if astrograph_output.frame_number[i]\
                        % astrograph_output.total_frames[i]\
                        == 0:

                    # Add the set of files in a single collect to
                    # the list of collects in this directory.
                    self.collect_files.append(files_processed)

                    # Reset the files that have been processed.
                    files_processed = []