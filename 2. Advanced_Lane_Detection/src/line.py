import cv2
import numpy as np


class Line:
    """"Lane line object"""

    def __init__(self):
        # indicator of whether the lane was detected in the last frame
        self.detected = False
        # polynomial coefficients of the last n fits of the line in pixel unit/meters
        self.recent_fitted_coeffs_in_pix = []
        self.recent_fitted_coeffs_in_meters = []
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # x values for detected line pixels
        self.x_pix_values = None
        # y values for detected line pixels
        self.y_pix_values = None
        # window margin attribute for ease of access
        self.window_margin = 50

    @property
    def radius_of_curvature_in_pixels(self):
        """ Property to get the radius of the curvature in pixels of the line """
        # get y value of at the bottom of the image
        y = np.max(self.y_pix_values)
        # find a, b, c coefficients by averating the past fitted coefficients
        a, b, c = np.mean(self.recent_fitted_coeffs_in_pix, axis=0)
        return ((1 + (2 * a * y + b) ** 2) ** 1.5) / np.absolute(2 * a)

    @property
    def radius_of_curvature_in_meters(self):
        """ Property to get the radius of the curvature in meters of the line """
        # get y value of at the bottom of the image
        y = np.max(self.y_pix_values)
        # convert coefficient values in pixels to meters
        a, b, c = np.mean(self.recent_fitted_coeffs_in_meters, axis=0)
        return ((1 + (2 * a * y + b) ** 2) ** 1.5) / np.absolute(2 * a)


class LaneDetection:
    """ Lane detection and drawing over the input warped image, use Line objects as left and right lanes """

    Y_METERS_PER_PIX = 30 / 720  # meters per pixel in y dimension
    X_METERS_PER_PIX = 3.7 / 700  # meters per pixel in x dimension

    def __init__(
        self,
        img,
        num_windows=9,
        window_margin=50,
        min_pixels=50,
        fit_tolerance=100,
        cache_size=15,
    ):
        self.img = img
        self.num_windows = num_windows  # number of sliding windows
        self.window_margin = window_margin  # width of sliding windows
        self.min_pixels = min_pixels  # minimum number of pixels found in a window
        self.fit_tolerance = fit_tolerance  # margin for refit of lane lines
        self.cache_size = cache_size  # number of previously fitted coefficients kept
        self.left_lane = Line()  # instantiate left and right lane objects
        self.right_lane = Line()

    def get_lane_start_location(self):
        """
        Obtained the approximate locations of lane lines using histogram peaks and sliding windows
        updates the line_base_pos attribute of left and right lane lines starting point of left and right lanes
        """
        # using histogram peaks of the bottom half of the image to find approximate location of the lanes
        histogram = np.sum(self.img[self.img.shape[0] // 2 :, :], axis=0)
        # the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        # set base positions of left and right lanes
        self.left_lane.line_base_pos = np.argmax(histogram[:midpoint])
        self.right_lane.line_base_pos = np.argmax(histogram[midpoint:]) + midpoint

    def get_line_sliding_windows(self):
        """
        Iterate through the sliding windows and track the line,
        used when no previous frames are avaliable,
        appends to left and right lanes' x_pix_values and y_pix_values attributes
        """
        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(self.img.shape[0] // self.num_windows)
        # Identify the x and y positions of all nonzero (i.e. activated) pixels in the image
        nonzero = self.img.nonzero()
        y = np.array(nonzero[0])
        x = np.array(nonzero[1])

        # use sliding window method to locate lane lines for both left and right lanes
        for lane in [self.left_lane, self.right_lane]:
            # initialize iterations parameters
            x_current = lane.line_base_pos
            lane_inds = []
            # iterate through the windows and track line
            for window in range(self.num_windows):
                # y coordinates for the current sliding window
                win_y_low = self.img.shape[0] - (window + 1) * window_height
                win_y_high = self.img.shape[0] - window * window_height
                # x coordinates for the current sliding window
                win_x_low = x_current - self.window_margin
                win_x_high = x_current + self.window_margin

                # Identify the nonzero pixels in x and y within the window i
                valid_inds = (
                    (y >= win_y_low)
                    & (y < win_y_high)
                    & (x >= win_x_low)
                    & (x < win_x_high)
                ).nonzero()[0]

                # append these indices to left and right lane indices
                lane_inds.append(valid_inds)
                # If found > min_pixels pixels, recenter next window on their mean position
                if len(valid_inds) > self.min_pixels:
                    x_current = np.int(np.mean(x[valid_inds]))

            # Extract lane line pixel positions and update x_pix_values, y_pix_values attributes of the line object
            try:
                # concat lane inds to make 1-d
                lane_inds = np.concatenate(lane_inds)
            except ValueError:
                # Avoids an error if the above is not implemented fully
                pass

            lane.x_pix_values, lane.y_pix_values = (
                x[lane_inds],
                y[lane_inds],
            )
            # update line status detected to be true
            lane.detected = True

    def get_line_search_prior(self):
        """
        Get lanes from fitting polynomials only when lanes are significantly different
        used when previous frames are avaliable,
        appends to left and right lanes' x_pix_values and y_pix_values attributes
        """
        # get non-zero elements for the image
        y, x = self.img.nonzero()
        y = np.array(y)
        x = np.array(x)

        # get fitted coefficients from previous fitted frames
        for lane in [self.left_lane, self.right_lane]:
            lane_inds = []
            a, b, c = lane.recent_fitted_coeffs_in_pix[-1]
            # Set the area of search based on nonzero x-values within the +/- tolerance of the polynomial function
            lane_inds.append(
                (x > (a * (y ** 2) + b * y + c - self.fit_tolerance))
                & (x < (a * (y ** 2) + b * y + c + self.fit_tolerance))
            )
            # update lanes x_pix_values, y_pix_values attributes
            lane.x_pix_values, lane.y_pix_values = (
                x[lane_inds],
                y[lane_inds],
            )

    def fit_polynomial_on_lanes(self):
        """
        Fit polynomial on lane lines in terms of pixel values and meters
        appends to left and right lanes' recent_fitted_coeffs_in_pix and recent_fitted_coeffs_in_meters attributes
        """
        # Fit second order polynomials, obtain 3 coefficients, append to fitted coeffs
        for lane in [self.left_lane, self.right_lane]:
            # in pixels
            lane.recent_fitted_coeffs_in_pix.append(
                np.polyfit(lane.y_pix_values, lane.x_pix_values, 2)
            )
            # in meters
            lane.recent_fitted_coeffs_in_meters.append(
                np.polyfit(
                    lane.y_pix_values * self.Y_METERS_PER_PIX,
                    lane.x_pix_values * self.X_METERS_PER_PIX,
                    2,
                )
            )

    def detect(self):
        """
        method that detect lines,
        if previously no line was detected, use sliding window method,
        else use search prior method

        """
        # get the lane lines starting locations using histogram peaks method
        self.get_lane_start_location()
        # if don't have lane lines info, use sliding window method
        if (self.left_lane.detected == False) or (self.right_lane.detected == False):
            self.get_line_sliding_windows()
        # if have lane lines info, use prior search method
        else:
            self.get_line_search_prior()
        # fit polynomials on left and right lane lines
        self.fit_polynomial_on_lanes()
        # create an annotated output image with left and right lanes colored
        img_fit = np.dstack((self.img, self.img, self.img)) * 255
        # left lane red
        img_fit[self.left_lane.y_pix_values, self.left_lane.x_pix_values] = [
            255,
            0,
            0,
        ]
        # right lane blue
        img_fit[self.right_lane.y_pix_values, self, self.right_lane.x_pix_values] = [
            0,
            0,
            255,
        ]
        return self.left_lane, self.right_lane, img_fit
