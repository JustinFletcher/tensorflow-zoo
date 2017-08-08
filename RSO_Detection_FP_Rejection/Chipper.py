

import numpy as np


class Chipper(object):

    """
    A class which instantiates objects that pull chips
    from images and image stacks. Call the instance to
    produce the chips.

    Properties:
        frame_stack: stack of frames in the collect
        centroid: mean RSO centroid
        width: chip width
        height: chip height
        chip_stack: stack of chips around the centroid
        label:
    """

    def __init__(self, frame_stack, centroid,
                 width, height):

        self.frame_stack = frame_stack

        self.centroid = centroid

        self.width = width

        self.height = height

        self.chip_stack = []

        self.chip_image_stack()

    @staticmethod
    def chip_image(image, chip_coords,
                   chip_width, chip_height):

        """
        Given an image stack, extract arbitrarily-many chips,
        as specified by the input chip_coord_list. Chips will
        center on the coordinates given in chip_coord_list,
        and will have extent specified by chip_width and
        chip_height.
        """

        # Compute the extent of the images.
        leftmost =\
            int(chip_coords[1] - np.round(chip_width / 2.0))
        rightmost =\
            int(chip_coords[1] + np.round(chip_width / 2.0))
        topmost =\
            int(chip_coords[0] - np.round(chip_height / 2.0))
        bottommost =\
            int(chip_coords[0] + np.round(chip_height / 2.0))

        # Adjust chip locations to handle edge cases.
        if leftmost < 0:

            rightmost += -leftmost
            leftmost = 0

        if topmost < 0:

            bottommost += -topmost
            topmost = 0

        if rightmost > image.shape[0]:

            leftmost -= rightmost - image.shape[0]
            rightmost = image.shape[0]

        if bottommost > image.shape[1]:

            topmost -= bottommost - image.shape[1]
            bottommost = image.shape[1]

        # Extract the chip from the image.
        chip = image[leftmost:rightmost, topmost:bottommost]

        return chip


    def chip_image_stack(self):

        for image in self.frame_stack:

            self.chip_stack.append(Chipper.chip_image(
                image, self.centroid, self.width, self.height))