

from ObjectAstrometry import ObjectAstrometry
import os


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


class AstroGraphOutput(object):

    """
    Class to parse a directory of .ast files or a list
    of .ast files. Properties are the same as those in
    the ObjectAstrometry class.
    """

    def __init__(self, dirPath, fileList=None):

        # Path to directory where the .ast files are
        # stored.
        self.dirPath = fix_path(dirPath)

        # List of .ast files. Only required if parsing
        # a defined list of .ast files.
        self.fileList = fileList

        # File name.
        self.file_name = []

        # Names of identified objects.
        self.Name = []

        # Locations of object centroids.
        self.object_centroid = []

        # Objects' full width at half max.
        self.FWHM = []

        # Objects' classification.
        self.Classification = []

        # Dates and times.
        self.Date_Time = []

        # Taskings.
        self.Tasking = []

        # Image RAs.
        self.Image_RA = []

        # Image decs.
        self.Image_Dec = []

        # Reference frames.
        self.RefFrame = []

        # Image mags.
        self.Image_Mag = []

        # Image mag errors.
        self.Image_MagErr = []

        # Filters.
        self.Filter = []

        # Image fluxes.
        self.Image_Flux = []

        # Image inst mags.
        self.Image_InstMag = []

        # Frame number.
        self.frame_number = []

        # Total frames.
        self.total_frames = []

        # Call to extract the detections from the
        # stack of .ast files.
        self.get_detections()

    def get_detections(self):

        # Continue through conditional if parsing
        # all files in a directory.
        if not self.fileList:

            # Initialize the list of .ast files.
            files = []

            # Avoid placing hidden directories in
            # the list of files.
            for file in os.listdir(self.dirPath):

                if not file.startswith('.'):

                    files.append(file)

            # Sort the files.
            files = sorted(files)

            # Pull the files from the directory that
            # have the .ast extension.
            for file in files:

                file_name, file_extension = \
                    os.path.splitext(file)

                # Parse the current .ast file.
                if file_extension == '.ast':

                    OA = ObjectAstrometry(self.dirPath + file)
                    self.file_name.append(file)
                    self.Name.append(OA.Name)
                    self.object_centroid.append(OA.object_centroid)
                    self.FWHM.append(OA.FWHM)
                    self.Classification.append(OA.Classification)
                    self.Date_Time.append(OA.Date_Time)
                    self.Tasking.append(OA.Tasking)
                    self.Image_RA.append(OA.Image_RA)
                    self.Image_Dec.append(OA.Image_Dec)
                    self.RefFrame.append(OA.RefFrame)
                    self.Image_Mag.append(OA.Image_Mag)
                    self.Image_MagErr.append(OA.Image_MagErr)
                    self.Filter.append(OA.Filter)
                    self.Image_Flux.append(OA.Image_Flux)
                    self.Image_InstMag.append(OA.Image_InstMag)
                    self.frame_number.append(OA.frame_number)
                    self.total_frames.append(OA.total_frames)

        # Continue through conditional if parsing a list of files.
        if self.fileList:
            
            for file in self.fileList:

                OA = ObjectAstrometry(self.dirPath + file)
                self.file_name.append(file)
                self.Name.append(OA.Name)
                self.object_centroid.append(OA.object_centroid)
                self.FWHM.append(OA.FWHM)
                self.Classification.append(OA.Classification)
                self.Date_Time.append(OA.Date_Time)
                self.Tasking.append(OA.Tasking)
                self.Image_RA.append(OA.Image_RA)
                self.Image_Dec.append(OA.Image_Dec)
                self.RefFrame.append(OA.RefFrame)
                self.Image_Mag.append(OA.Image_Mag)
                self.Image_MagErr.append(OA.Image_MagErr)
                self.Filter.append(OA.Filter)
                self.Image_Flux.append(OA.Image_Flux)
                self.Image_InstMag.append(OA.Image_InstMag)
                self.frame_number.append(OA.frame_number)
                self.total_frames.append(OA.total_frames)