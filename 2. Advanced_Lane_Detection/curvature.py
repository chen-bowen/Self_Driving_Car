import cv2
import numpy as np


class CurvatureDetection:
    """ Lane detection and drawing over the input warped image """

    def __init__(self, num_windows=9, window_margin=100, min_pixels=50, fit_margin=100):
        self.num_windows = num_windows  # number of sliding windows
        self.window_margin = window_margin  # width of sliding windows
        self.min_pixels = min_pixels  # minimum number of pixels found in a window
        self.fit_margin = fit_margin  # margin for refit of lane lines
        self.left_fit_prior = None  # used for saving fit for previous frames
        self.right_fit_prior = None

    def get_lane_approx_location(self, img):
        """
        Obtained the approximate locations of lane lines using histogram peaks and sliding windows
        Returns: starting point of left and right lanes
        """
        # using histogram peaks of the bottom half of the image to find approximate location of the lanes
        histogram = np.sum(img[img.shape[0] // 2 :, :], axis=0)
        # the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        return leftx_base, rightx_base

   def get_line_sliding_windows(self, img):
        """ Iterate through the sliding windows and track the line, used when no previous frames are avaliable """
        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(img.shape[0] // self.num_windows)
        # Identify the x and y positions of all nonzero (i.e. activated) pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # initialize iterations parameters
        leftx_current = leftx_base
        rightx_current = rightx_base
        left_lane_inds = []
        right_lane_inds = []
        # iterate through the windows and track line
        for window in range(self.num_windows):
            # y coordinates for the current sliding window
            # (only has 2 values since left right lanes are the same)
            win_y_low = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - window * window_height
            # x coordinates for the current sliding window
            # has 4 values (high, low) x (left, right)
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Identify the nonzero pixels in x and y within the window i
            valid_left_inds = (
                (nonzeroy >= win_y_low)
                & (nonzeroy < win_y_high)
                & (nonzerox >= win_xleft_low)
                & (nonzerox < win_xleft_high)
            ).nonzero()[0]
            valid_right_inds = (
                (nonzeroy >= win_y_low)
                & (nonzeroy < win_y_high)
                & (nonzerox >= win_xright_low)
                & (nonzerox < win_xright_high)
            ).nonzero()[0]
            # append these indices to left and right lane indices
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > min_pixels pixels, recenter next window on their mean position
            if len(valid_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[valid_left_inds]))
            if len(valid_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[valid_right_inds]))

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty

    def get_line_search_prior(self, img):
        """ Get lanes from fitting polynomials only when lanes are significantly different """
        # get non-zero elements for the image
        nonzeroy, nonzerox = img.nonzero()
        nonzeroy = np.array(nonzeroy)
        nonzerox = np.array(nonzerox)

        # get fitted coefficients from previous fitted frames
        left_a, left_b, left_c = self.left_fit_prior
        right_a, right_b, right_c = self.right_fit_prior

        # Set the area of search based on nonzero x-values within the +/- margin of the polynomial function
        left_lane_inds = (
            nonzerox > (left_a * (nonzeroy ** 2) + left_b * nonzeroy + left_c - margin)
        ) & (
            nonzerox < (left_a * (nonzeroy ** 2) + left_b * nonzeroy + left_c + margin)
        )

        right_lane_inds = (
            nonzerox
            > (right_a * (nonzeroy ** 2) + right_b * nonzeroy + right_c - margin)
        ) & (
            nonzerox
            < (right_a * (nonzeroy ** 2) + right_b * nonzeroy + right_c + margin)
        )
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty

    def fit_polynomial_on_lanes(self, img, leftx, lefty, rightx, righty):
        """Fit polynomial on lane lines"""
        # get image shapes
        img_shape = img.shape
        # Fit second order polynomials, obtain 3 coefficients
        left_a, left_b, left_c = np.polyfit(lefty, leftx, 2)
        right_a, right_b, right_c = np.polyfit(righty, rightx, 2)
        # get x and y values
        y = np.linspace(0, img_shape[0] - 1, img_shape[0])
        x_left = left_a * y ** 2 + left_b * y + left_c
        x_right = right_a * y ** 2 + right_b * y + right_c
        # update the prior
        self.left_fit_prior = left_a, left_b, left_c
        self.right_fit_prior = right_a, right_b, right_c

        return x_left, x_right, y


    # def get_lane_curvature(self, )