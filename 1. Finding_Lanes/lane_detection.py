import numpy as np
from utils import (
    hls_scale,
    define_region_vertices,
    get_lanes_segments,
    get_line_end_pts,
    weighted_img,
)
import cv2


class LaneDetection:

    LANE_LINE_IMG_RATIO = 0.63

    def __init__(
        self,
        cache_size=5,
        gaussian_blur_params={"kernel_size": 3},
        color_filter_params={
            "white_bounds": [np.uint8([0, 150, 0]), np.uint8([255, 255, 255])],
            "yellow_bounds": [np.uint8([20, 0, 100]), np.uint8([50, 255, 255])],
        },
        region_filter_params={"ratios": [0.48, 0.55, 0.51, 0.55]},
        canny_params={"thresholds": [100, 200]},
        slope_params={"bounds": [0.5, 1]},
        hough_params={
            "rho": 1,
            "theta": np.pi / 180,
            "threshold": 35,
            "min_line_len": 5,
            "max_line_gap": 2,
        },
        prob_smoothing_params={"current_line_weight": 0.4, "past_lines_weight": 0.6},
    ):
        self.cache_size = cache_size
        self.gaussian_blur_params = gaussian_blur_params
        self.region_filter_params = region_filter_params
        self.color_filter_params = color_filter_params
        self.canny_params = canny_params
        self.slope_params = slope_params
        self.hough_params = hough_params
        self.prob_smoothing_params = prob_smoothing_params
        self.init_cache()

    def init_cache(self):
        """ Initialize left and right lane cache """
        self.left_lane_cache = list()
        self.right_lane_cache = list()

    def probablistic_smoothing(self, line, cache):
        """ take the average of past cached endpoints as the new end point of line segments"""

        new_line_wght = self.prob_smoothing_params["current_line_weight"]
        past_lines_wght = self.prob_smoothing_params["past_lines_weight"]

        if line is None:
            return np.mean(cache, axis=0).round().astype(int), cache[-self.cache_size :]

        smoothed_line = (
            (
                (
                    new_line_wght * np.array(line)
                    + past_lines_wght * np.mean(cache, axis=0)
                )
                .round()
                .astype(int)
            )
            if len(cache) > 0
            else np.array(line)
        )

        cache.append(line)

        return smoothed_line, cache[-self.cache_size :]

    def color_filter(self, image):
        """ Create a color mask that only persists yellow and white color """
        converted = hls_scale(image)

        # white color mask
        white_lower_bnd, white_upper_bnd = self.color_filter_params["white_bounds"]
        white_mask = cv2.inRange(converted, white_lower_bnd, white_upper_bnd)

        # yellow color mask
        yellow_lower_bnd, yellow_upper_bnd = self.color_filter_params["yellow_bounds"]
        yellow_mask = cv2.inRange(converted, yellow_lower_bnd, yellow_upper_bnd)

        # combine the mask
        mask = cv2.bitwise_or(white_mask, yellow_mask)

        return cv2.bitwise_and(image, image, mask=mask)

    def gaussian_blur(self, img):
        """Applies a Gaussian Noise kernel"""
        kernel_size = self.gaussian_blur_params["kernel_size"]
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def region_of_interest(self, img):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        `vertices` should be a numpy array of integer points.
        """
        # get region vertices
        r1, r2, r3, r4 = self.region_filter_params["ratios"]
        img_height, img_width = img.shape
        vertices = define_region_vertices(img_height, img_width, r1, r2, r3, r4)

        # defining a blank mask to start with
        mask = np.zeros_like(img)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, [vertices], ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def canny(self, img):
        """Applies the Canny transform"""
        low_threshold, high_threshold = self.canny_params["thresholds"]
        return cv2.Canny(img, low_threshold, high_threshold)

    def get_lane_lines(self, lines, image):
        """
        Given a set of lines (both sloped left and right), return the end points of both
        left and right sloped lines

        return: left_lane, right_lane (slope, intercept)
        """
        # get image shape
        img_height, img_width, _ = image.shape

        # get left and right lanes
        slope_lower_bnd, slope_upper_bnd = self.slope_params["bounds"]
        left_lane, right_lane = get_lanes_segments(
            lines, slope_lower_bnd, slope_upper_bnd
        )

        # use get_line_end_pts to convert left_lane and right_lane to their corresponding end points
        left_lane = get_line_end_pts(
            left_lane, y1=img_height, y2=self.LANE_LINE_IMG_RATIO * img_height
        )
        right_lane = get_line_end_pts(
            right_lane, y1=img_height, y2=self.LANE_LINE_IMG_RATIO * img_height
        )

        # update cache and apply probablistic smoothing
        left_lane, self.left_lane_cache = self.probablistic_smoothing(
            left_lane, self.left_lane_cache
        )

        right_lane, self.right_lane_cache = self.probablistic_smoothing(
            right_lane, self.right_lane_cache
        )

        return left_lane, right_lane

    def draw_lines(self, img, lines, color=[255, 0, 0], thickness=5):
        """
        Use get_lane_lines to draw complete lines over image
        """
        # draw left and right lane lines
        for x1, y1, x2, y2 in self.get_lane_lines(lines, img):
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    def hough_lines(self, img):
        """
        `img` should be the output of a Canny transform.

        Returns an image with hough lines drawn.
        """
        # get parameters
        rho = self.hough_params["rho"]
        theta = self.hough_params["theta"]
        threshold = self.hough_params["threshold"]
        min_line_len = self.hough_params["min_line_len"]
        max_line_gap = self.hough_params["max_line_gap"]

        lines = cv2.HoughLinesP(
            img,
            rho,
            theta,
            threshold,
            np.array([]),
            minLineLength=min_line_len,
            maxLineGap=max_line_gap,
        )

        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

        # draw lines over image
        self.draw_lines(line_img, lines)
        return line_img

    def detect(self, img):
        """ Combine all the previous steps and perform lane detection on given image """
        # 1. color filter
        lane_img = self.color_filter(img.copy())
        # 2. gaussian blur
        lane_img = self.gaussian_blur(lane_img)
        # 3.canny edge detection
        lane_img = self.canny(lane_img)
        # 4. region of interest crop
        lane_img = self.region_of_interest(lane_img)
        # 5. hough lines
        lane_img = self.hough_lines(lane_img)
        # 6. overlay lane over original image
        result_img = weighted_img(lane_img, img)

        return result_img
