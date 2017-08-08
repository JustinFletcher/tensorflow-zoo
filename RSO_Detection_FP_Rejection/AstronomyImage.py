

from astropy.io import fits
import numpy as np


class AstronomyImage(object):

    """
    A class which parses a .fits file.

    Properties:
        file_path: path to the .fits file
        image: image pulled from the .fits file
        header: headers information pulled from the
            .fits file
    """

    def __init__(self, file_path):

        # Path to .fits file.
        self.file_path = file_path

        # Image in .fits file.
        self.image = []

        # Header in .fits file.
        self.header = []

        # Call to parse the .fits file.
        self.parse_fits()

    def parse_fits(self):

        hdulist = fits.open(self.file_path)
        hdu = hdulist[0]
        self.image = np.asarray(hdu.data, 'float32')
        self.header = hdu.header
