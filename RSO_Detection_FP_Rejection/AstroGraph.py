

import csv


class AstroGraph(object):

    """
    A class that wraps AstroGraph.

    Properties:
        Various AstroGraph properties. Add properties as needed.
    """

    def __init__(self,
                 instance_ID=
                 'DOTS-System',
                 AstroGraph_sensor_searchDirectory=
                 '/home/jermws/Unclass_Sensor/',
                 sensor_name='RME02.sensor',
                 AstroGraph_match_brightStarLimit=-3,
                 AstroGraph_starFit_photometryMethod=
                 'Fixed /home/jermws/Unclass_Sample_Data/',
                 starDetect_threshold=2.5,
                 AstroGraph_targetDetect_threshold=4.2,
                 AstroGraph_targetDetect_edgeLimits=
                 '{10,10}{10,10}',
                 AstroGraph_targetDetect_sizeLimits='{3,*}',
                 AstroGraph_satelliteCatalog_tlePath=
                 '/home/jermws/Unclass_Elset/catalog_3l_2016_11_17_am.txt',
                 correct_darkFile=
                 '/home/ravenuser/data/DARK_2x2_NormalMode_F00349.fit',
                 correct_flatFile=
                 '/home/ravenuser/data/FLAT_2x2_NormalMode_F00349.fits',
                 starCatalog_path=
                 '/home/jermws/AstroGraph/SSTRC4',
                 AstroGraph_data_rootPath=
                 '/data/remote'):

        # Instance or simulation id of the group for
        # this service membership.
        self.instance_id = instance_ID

        # Directory to recursively search for sensor
        # property files.
        self.AstroGraph_sensor_searchDirectory =\
            AstroGraph_sensor_searchDirectory

        # Sensor name
        self.sensor_name = sensor_name

        # Magnitude limit for the brightest star in
        # the FOV for skipped frames. Set to -30.0
        # for dynamic sensor estimate.
        self.AstroGraph_match_brightStarLimit =\
            AstroGraph_match_brightStarLimit

        # Photometric fit method to use
        self.AstroGraph_starFit_photometryMethod =\
            AstroGraph_starFit_photometryMethod

        # Star detection threshold in sigma above noise
        self.starDetect_threshold = starDetect_threshold

        # Target detection threshold in sigma
        self.AstroGraph_targetDetect_threshold =\
            AstroGraph_targetDetect_threshold

        # Valid detected target proximity limits to
        # sensor edge in pixels.
        self.AstroGraph_targetDetect_edgeLimits =\
            AstroGraph_targetDetect_edgeLimits

        # Valid detected target size limits in pixels.
        self.AstroGraph_targetDetect_sizeLimits =\
            AstroGraph_targetDetect_sizeLimits

        # Valid detected target size limits in pixels.
        self.AstroGraph_satelliteCatalog_tlePath =\
            AstroGraph_satelliteCatalog_tlePath

        # Path to dark correction image.
        self.correct_darkFile = correct_darkFile

        # Path to flat field correction image.
        self.correct_flatFile = correct_flatFile

        # Path to the astrometric/photometric star
        # catalog.
        self.starCatalog_path = starCatalog_path

        # Root path for reading data.
        self.AstroGraph_data_rootPath =\
            AstroGraph_data_rootPath

        # Create dictionary mapping user defined
        # language to command line options.
        with open('AstroGraph_attributes_and_commands.csv', 'r')\
                as f:

            reader = csv.reader(f)
            self.command_line_dict =\
                {row[0]: row[1] for row in reader}

        # Create dictionary mapping user defined
        # language to parameter values.
        self.parameter_value_dict =\
            {'help': '',
             'instance ID': instance_ID,
             'sensor': sensor_name,
             'sensor directory':
                 AstroGraph_sensor_searchDirectory,
             'batch': '',
             'star detection threshold':
                 starDetect_threshold,
             'target detection threshold':
                 AstroGraph_targetDetect_threshold,
             'target detection edge limits':
                 AstroGraph_targetDetect_edgeLimits,
             'target detection size limits':
                 AstroGraph_targetDetect_sizeLimits,
             'bright star limit':
                 AstroGraph_match_brightStarLimit,
             'star fit method':
                 AstroGraph_starFit_photometryMethod,
             'star catalog': starCatalog_path,
             'satellite catalog TLE path':
                 AstroGraph_satelliteCatalog_tlePath,
             'correct dark file': correct_darkFile,
             'correct flat file': correct_flatFile,
             'data root path': AstroGraph_data_rootPath}

    def generate_commandline(self, commands):

        """
        Generates an AstroGraph command line to perform a
        particular task based on input commands.
        """

        # Initialize the command line to be generated.
        command_line = './AstroGraph '

        # Iterate over each command, appending it and
        # the corresponding parameter to the command
        # string.
        for _, command in enumerate(commands):

            command_line += ' '\
                            + self.command_line_dict[command]\
                            + ' '\
                            + str(
                self.parameter_value_dict[command])

        return command_line

    @staticmethod
    def run_properties_file():

        """
        Run an AstroGraph properties file.
        """

        return './AstroGraph'

    @staticmethod
    def list_properties():

        """
        List all of the AstroGraph properties.
        """

        return './AstroGraph --list-properties'

    @staticmethod
    def list_commands():

        """
        List all of the AstroGraph commands.
        """

        return './AstroGraph --help'