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

    def abs_sobel_thresh(self, img, orient="x"):
        """ produce a absoulte threshold binary filter for directional gradient """
        # x, y directions
        orient_dir = {"x": [1, 0], "y": [0, 1]}
        # convert grayscale
        grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # calculate gradient or a certain orient
        derivative = cv2.Sobel(gray, cv2.CV_64F, *orient_dir[orient])
        abs_derivative = np.absolute(derivative)
        # Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sabel = np.uint8(255 * abs_derivative / np.max(abs_derivative))
        # return mask
        thresh_min, thresh_max = self.sobel_threshold
        grad_binary = np.zeros_like(scaled_sabel)
        grad_binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

        return grad_binary

    def mag_thresh(self, image):
        """ produce a magitude threshold binary filter for directional gradient """
        # x, y directions
        orient_dir = {"x": [1, 0], "y": [0, 1]}
        # convert grayscale
        grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # calculate gradient or a certain orient
        x_derivative = cv2.Sobel(gray, cv2.CV_64F, *orient_dir["x"])
        y_derivative = cv2.Sobel(gray, cv2.CV_64F, *orient_dir["y"])
        # calculate gradient magnitude
        derivative_magnitude = np.sqrt(x_derivative ** 2 + y_derivative ** 2)
        # Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sabel = np.uint8(
            255 * derivative_magnitude / np.max(derivative_magnitude)
        )
        # return mask
        thresh_min, thresh_max = self.sobel_threshold
        mag_binary = np.zeros_like(scaled_sabel)
        mag_binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

        return mag_binary

    def dir_threshold(self, image):
        # x, y directions
        orient_dir = {"x": [1, 0], "y": [0, 1]}
        # convert grayscale
        grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # calculate gradient or a certain orient
        x_derivative = cv2.Sobel(gray, cv2.CV_64F, *orient_dir["x"])
        y_derivative = cv2.Sobel(gray, cv2.CV_64F, *orient_dir["y"])
        # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        # return mask
        thresh_min, thresh_max = self.sobel_threshold
        mag_binary = np.zeros_like(absgraddir)
        mag_binary[(absgraddir >= thresh_min) & (absgraddir <= thresh_max)] = 1
        return dir_binary

    def apply_gradient_filter(self, image):
        """ Combine the 3 different thresholds """
        # get the 3 different thresholds
        gradx = self.abs_sobel_thresh(image, orient="x")
        grady = self.abs_sobel_thresh(image, orient="y")
        mag_binary = self.mag_thresh(image)
        dir_binary = self.dir_threshold(image)

        # combine filters
        combined = np.zeros_like(dir_binary)
        combined[
            ((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))
        ] = 1
        return combined


class ColorFiltering:
    """ Applies color filtering over a given image """

    def __init__(
        self,
        white_bounds={"lower": [], "upper": []},
        yellow_bounds={"lower": [], "upper": []},
    ):
        self.white_bounds = white_bounds
        self.yellow_bounds = yellow_bounds

    def hls_scale(self, img):
        """ Applies the HLS transform """
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    def white_mask(self, converted_img):
        """ color masking for white """
        # white color mask
        lower = np.uint8(self.white_bounds["lower"])
        upper = np.uint8(self.white_bounds["upper"])
        white_mask = cv2.inRange(converted_img, lower, upper)
        return white_mask

    def yellow_mask(self, converted_img):
        """ color masking for white """
        # yellow color mask
        lower = np.uint8(self.yellow_bounds["lower"])
        upper = np.uint8(self.yellow_bounds["upper"])
        yellow_mask = cv2.inRange(converted_img, lower, upper)
        return white_mask

    def apply_color_filter(self, img):
        """ Applies the white and yellow color mask over image """
        # Convert image into hls scale
        converted = self.hls_scale(img)
        # get yellow and white mask
        white_mask = self.white_mask(converted)
        yellow_mask = self.yellow_mask(converted)
        # combine the mask
        mask = cv2.bitwise_or(white_mask, yellow_mask)
        # apply the mask over the image
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        return masked_img
