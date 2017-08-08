

class ObjectAstrometry(object):

    """
    Class to parse an .ast file. Class attributes
    are equivalent to the parameter names in the
    .ast file. Will parse either all of the .ast
    files in a directory or a list of .ast files
    provided by the user.
    """

    def __init__(self, file_path):

        # Path to .ast file.
        self.file_path = file_path

        # Name of identified object.
        self.Name = []

        # Location of object centroid.
        self.object_centroid = []

        # Object full width at half max.
        self.FWHM = []

        # Object classification.
        self.Classification = []

        # Date and time.
        self.Date_Time = []

        # Tasking.
        self.Tasking = []

        # Image RA.
        self.Image_RA = []

        # Image dec.
        self.Image_Dec = []

        # Reference frame.
        self.RefFrame = []

        # Image mag.
        self.Image_Mag = []

        # Image mag error.
        self.Image_MagErr = []

        # Filter.
        self.Filter = []

        # Image flux.
        self.Image_Flux = []

        # Image inst mag.
        self.Image_InstMag = []

        # Call to parse the .ast file.
        self.parse_ast()

    def parse_ast(self):

        # Open the .ast file.
        file = open(self.file_path)

        # Read all of the file lines into a list.
        file_lines = file.readlines()

        # Assume there are no detections in the file.
        number_of_detections = 0
        detections_file = []

        for i in range(len(file_lines)):

            # Pull lines from file in sequential order.
            line = file_lines[i]

            # Continue through the conditional if this line
            # is a header line.
            if line[0] is '#':

                # Colon delimit the line.
                line_split = line.split(':')

                # Delimit the header line object name.
                line_subsplit = line_split[0].split('# ')

                # Continue through the conditional if this
                # line contains frame number information.
                if line_subsplit[1] == 'Image group':

                    # Get the frame data.
                    frame_data = line_split[1]

                    # Define the frame numbering corresponding
                    # to this .ast file.
                    self.frame_number = int(frame_data[2])

                    # Define the total number of frames.
                    self.total_frames = int(frame_data[4])

        for i in range(len(file_lines)):

            if file_lines[i][:-1] == '[Object Astrometry]':

                # Pull all of the detections in this .ast
                # file.
                detections_file = file_lines[i + 7:]

                # Compute the number of detections in this
                # .ast file.
                number_of_detections = int(len(detections_file)
                                           / 6)

                break

        if number_of_detections:

            # Pull the attributes from the .ast file
            # using prior knowledge of the file structure.
            for i in range(number_of_detections):

                # Extract one detection from the total
                # number of detections in the file.
                detection = detections_file[i * 6 + 1:(i + 1)
                                                      * 6]

                for line in range(len(detection)):

                    # Space delimit for each line of the detection.
                    line_entries = detection[line].split(' ')

                    # Remove empty spaces.
                    line_entries = list(filter(None, line_entries))

                    # Assign attribute values using prior knowledge
                    # of the file structure.
                    if line is 0:

                        self.Name.append(line_entries[0])
                        self.object_centroid.append(
                            [float(line_entries[1]),
                             float(line_entries[2])])
                        self.FWHM.append(float(line_entries[3]))
                        self.Classification.append(line_entries[4])

                    elif line is 1:

                        self.Date_Time.append([line_entries[1],
                                               line_entries[2]])
                        self.Tasking.append(line_entries[3] +
                                            ' ' + line_entries[4])

                    elif line is 2:

                        self.Image_RA.append(line_entries[1])
                        self.Image_Dec.append(line_entries[2])
                        self.RefFrame.append(line_entries[3])

                    elif line is 3:

                        self.Image_Mag.append(float(line_entries[1]))
                        self.Image_MagErr.append(
                            float(line_entries[2]))
                        self.Filter.append(line_entries[3])

                    else:

                        self.Image_Flux.append(float(line_entries[1]))
                        self.Image_InstMag.append(
                            float(line_entries[2]))

        else:

            self.Name.append(None)
            self.object_centroid.append(None)
            self.FWHM.append(None)
            self.Classification.append(None)
            self.Date_Time.append(None)
            self.Tasking.append(None)
            self.Image_RA.append(None)
            self.Image_Dec.append(None)
            self.RefFrame.append(None)
            self.Image_Mag.append(None)
            self.Image_MagErr.append(None)
            self.Filter.append(None)
            self.Image_Flux.append(None)
            self.Image_InstMag.append(None)