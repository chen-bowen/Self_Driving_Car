from src.perspective import Calibration
from src.line import LaneDetection
from src.filters import ColorFiltering, GradientFiltering, MorphologyFiltering
from src.annotate import AnnotateFrame
import numpy as np
import cv2
import glob


def process_frame(
    img,
    gradient_params,
    color_params,
    morph_params,
    lane_detect_params,
    annotate_params,
):
    """ pipeline that process and annotate a single frame """

    # apply color filters
    color = ColorFiltering(**color_params)
    color_filtered_img = color.apply_color_filter(img)

    # apply gradient filters
    gradient = GradientFiltering(**gradient_params)
    gradient_filtered_img = gradient.apply_gradient_filter(img)

    # combine gradient and color filter
    combined_filtered_image = np.logical_or(color_filtered_img, gradient_filtered_img)

    # apply morphology filter
    morph = MorphologyFiltering(**morph_params)
    moprhed_image = morph.apply_morphology_filter(combined_filtered_image)

    # calibrate the carmera
    perspective_transformer = Calibration()
    perspective_transformer.set_calibration()
    # undistort and convert image to bird's eye view
    birdeye_filtered_img = perspective_transformer.undistort_and_birdeye_transform(
        moprhed_image
    )
    birdeye_original_img = perspective_transformer.undistort_and_birdeye_transform(img)

    # detect lanes
    left_lane, right_lane, img_fit = LaneDetection(
        img=birdeye_filtered_img, **lane_detect_params
    ).detect()
    # annotate frame
    annotate = AnnotateFrame(
        left_lane,
        right_lane,
        img_assets=[
            birdeye_original_img,
            moprhed_image,
            birdeye_filtered_img,
            img_fit,
        ],
        perspective_transformer=perspective_transformer,
        **annotate_params,
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
                sobel_threshold=(50, 150),
                magnitude_threshold=(30, 100),
                direction_threshold=(0.7, 1.3),
            ),
            color_params=dict(s_thresholds=(150, 255), r_thresholds=(200, 255)),
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
