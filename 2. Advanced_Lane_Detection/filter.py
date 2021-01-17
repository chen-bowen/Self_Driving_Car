import cv2


class EdgeFiltering:
    """ applies combined edge filter for a given image """

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

    def combined_threshold(self, image):
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
