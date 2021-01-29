import cv2
import numpy as np


class GradientFiltering:
    """ applies combined gradient filter for a given image """

    def __init__(
        self,
        sobel_kernel_size=3,
        sobel_threshold=(0, 255),
        magnitude_threshold=(0, 255),
        direction_threshold=(0, np.pi / 2),
    ):
        self.sobel_kernel_size = sobel_kernel_size
        self.sobel_threshold = sobel_threshold
        self.magnitude_threshold = magnitude_threshold
        self.direction_threshold = direction_threshold

    def abs_sobel_thresh(self, img, orient="x"):
        """ produce a absolute threshold binary filter for directional gradient """
        # x, y directions
        orient_dir = {"x": [1, 0], "y": [0, 1]}
        # convert grayscale
        grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # calculate gradient or a certain orient
        derivative = cv2.Sobel(grayscale, cv2.CV_64F, *orient_dir[orient])
        abs_derivative = np.absolute(derivative)
        # Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255 * abs_derivative / np.max(abs_derivative))
        # return mask
        thresh_min, thresh_max = self.sobel_threshold
        _, grad_binary = cv2.threshold(
            scaled_sobel, thresh_min, thresh_max, cv2.THRESH_BINARY
        )

        return grad_binary.astype(bool)

    def mag_threshold(self, img):
        """ produce a magitude threshold binary filter for directional gradient """
        # x, y directions
        orient_dir = {"x": [1, 0], "y": [0, 1]}
        # convert grayscale
        grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # calculate gradient or a certain orient
        x_derivative = cv2.Sobel(grayscale, cv2.CV_64F, *orient_dir["x"])
        y_derivative = cv2.Sobel(grayscale, cv2.CV_64F, *orient_dir["y"])
        # calculate gradient magnitude
        derivative_magnitude = np.sqrt(x_derivative ** 2 + y_derivative ** 2)
        # Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(
            255 * derivative_magnitude / np.max(derivative_magnitude)
        )
        # return mask
        thresh_min, thresh_max = self.magnitude_threshold

        _, mag_binary = cv2.threshold(
            scaled_sobel, thresh_min, thresh_max, cv2.THRESH_BINARY
        )

        return mag_binary.astype(bool)

    def dir_threshold(self, img):
        # x, y directions
        orient_dir = {"x": [1, 0], "y": [0, 1]}
        # convert grayscale
        grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # calculate gradient or a certain orient
        x_derivative = cv2.Sobel(grayscale, cv2.CV_64F, *orient_dir["x"])
        y_derivative = cv2.Sobel(grayscale, cv2.CV_64F, *orient_dir["y"])
        # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
        absgraddir = np.arctan2(np.absolute(y_derivative), np.absolute(x_derivative))
        # return mask
        thresh_min, thresh_max = self.direction_threshold

        _, dir_binary = cv2.threshold(
            absgraddir, thresh_min, thresh_max, cv2.THRESH_BINARY
        )

        return dir_binary

    def apply_gradient_filter(self, image):
        """ Combine the 3 different thresholds """
        # get the 3 different thresholds
        gradx = self.abs_sobel_thresh(image, orient="x")
        grady = self.abs_sobel_thresh(image, orient="y")
        mag_binary = self.mag_threshold(image)
        dir_binary = self.dir_threshold(image)

        # combine filters
        gradient_filters = np.logical_and(gradx, grady)
        mag_dir_filters = np.logical_and(mag_binary, dir_binary)
        combined_filters = np.logical_or(gradient_filters, mag_dir_filters)
        return combined_filters.astype(np.uint8)


class ColorFiltering:
    """ Applies color filtering over a given image """

    def __init__(self, s_thresholds=(150, 255), r_thresholds=(200, 255)):
        self.s_thresholds = s_thresholds
        self.r_thresholds = r_thresholds

    def hls_scale(self, img):
        """ Applies the HLS transform """
        return cv2.cvtColor(img.copy(), cv2.COLOR_RGB2HLS)

    def s_channel_filter(self, img):
        """ color filter for s channel """
        # convert image to hls scale
        converted_img = self.hls_scale(img)
        # get threshold min and max
        thresh_min, thresh_max = self.s_thresholds
        # get s channel binary
        _, s_binary = cv2.threshold(
            converted_img[:, :, 2], thresh_min, thresh_max, cv2.THRESH_BINARY
        )
        return s_binary.astype(np.uint8)

    def r_channel_filter(self, img):
        """ color filter for s channel """
        # get threshold min and max
        thresh_min, thresh_max = self.r_thresholds
        # get s channel binary
        _, s_binary = cv2.threshold(
            img[:, :, 0], thresh_min, thresh_max, cv2.THRESH_BINARY
        )
        return s_binary.astype(np.uint8)

    def apply_color_filter(self, img):
        """ Applies the white and yellow color mask over image """
        # s channel filter
        s_filter = self.s_channel_filter(img)
        # r channel filter
        r_filter = self.r_channel_filter(img)
        # apply the mask over the image
        combined_filter = np.logical_or(s_filter, r_filter)
        return combined_filter
