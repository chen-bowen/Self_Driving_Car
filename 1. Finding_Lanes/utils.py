import cv2
import numpy as np


def hls_scale(img):
    """Applies the HLS transform"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)


def define_region_vertices(img_height, img_width, r1, r2, r3, r4):
    """ define region of interest using the ratios in region_filter_params """

    region_vertices = np.array(
        [
            [int(r1 * img_width), int(r2 * img_height)],
            [int(r3 * img_width), int(r4 * img_height)],
            [0, img_height],
            [img_width, img_height],
        ],
        dtype=np.int32,
    )
    return region_vertices


def get_lanes_segments(lines, lower_bnd, upper_bnd):
    """ Get left and right lane slopes and intercepts given a set of lines """
    # gather left sloped and right sloped lines in format of (slope, intercept)
    lft_lanes = []
    rht_lanes = []

    # assign weights to line segments of different lengths
    lft_weights = []
    rht_weights = []

    # if the slope is negative and between threholds, it's a left lane line
    # if the slope is postive and between thresholds, it's a right lane line
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # save slope, inter
        slope = (y2 - y1) / (x2 - x1)

        intercept = y1 - slope * x1
        seg_length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

        if lower_bnd <= slope <= upper_bnd:
            rht_lanes.append((slope, intercept))
            rht_weights.append(seg_length)

        elif -upper_bnd <= slope <= -lower_bnd:
            lft_lanes.append((slope, intercept))
            lft_weights.append(seg_length)

    # find weighted average of left and right lanes
    left_lane = (
        np.dot(lft_weights, lft_lanes) / np.sum(lft_weights)
        if len(lft_weights) > 0
        else None
    )
    right_lane = (
        np.dot(rht_weights, rht_lanes) / np.sum(rht_weights)
        if len(rht_weights) > 0
        else None
    )

    return left_lane, right_lane


def get_line_end_pts(line_segment, y1, y2):
    """ Convert line segment of slope and intercept to line end points"""
    if line_segment is None:
        return None

    slope, intercept = line_segment

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)

    return x1, y1, x2, y2


def weighted_img(img, initial_img, α=0.8, β=1.0, γ=0.0):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)