from src.perspective import Calibration
from src.line import LaneDetection
from src.filters import ColorFiltering, GradientFiltering
from src.annotate import AnnotateFrame
import numpy as np
import cv2
import glob


def process_frame(
    img, gradient_params, color_params, lane_detect_params, annotate_params
):
    """ pipeline that process and annotate a single frame """
    # apply gradient filters
    gradient = GradientFiltering(**gradient_params)
    binary_img = gradient.apply_gradient_filter(img)
    # apply color filters
    color = ColorFiltering(**color_params)
    color_filtered_img = color.apply_color_filter(binary_img)
    # calibrate the carmera
    perspective_transformer = Calibration()
    perspective_transformer.set_calibration()
    # undistort and convert image to bird's eye view
    birdeye_filtered_img = perspective_transformer.undistort_and_birdeye_transform(
        color_filtered_img
    )
    birdeye_original_img = perspective_transformer.undistort_and_birdeye_transform(img)

    # detect lanes
    left_lane, right_lane = LaneDetection(
        img=birdeye_filtered_img, **lane_detect_params
    )
    # annotate frame
    annotate = AnnotateFrame(
        left_lane,
        right_lane,
        img_assets=[
            birdeye_original_img,
            binary_img,
            birdeye_original_img,
        ],  # to do: add image fit
        **annotate_params
    )
    # produce the final blended frame
    res_frame = annotate.blend_frame()

    return res_frame


if __name__ == "__main__":
    # process all images in test folder
    for test_img in glob.glob("test_images/*.jpg"):
        img = cv2.imread(test_img)
        process_frame(
            img,
            gradient_params=dict(
                sobel_kernel_size=3,
                sobel_threshold=(0, 255),
                magnitude_threshold=(0, 255),
                direction_threshold=(0, np.pi / 2),
            ),
            color_params=dict(
                white_bounds={"lower": [0, 150, 0], "upper": [255, 255, 255]},
                yellow_bounds={"lower": [50, 0, 100], "upper": [100, 255, 255]},
            ),
            lane_detect_params=dict(
                num_windows=9,
                window_margin=50,
                min_pixels=50,
                fit_tolerance=100,
                cache_size=15,
            ),
            annotate_params=dict(
                line_width=50,
                lane_color=(255, 0, 0),
                road_region_color=(0, 255, 0),
            ),
        )
