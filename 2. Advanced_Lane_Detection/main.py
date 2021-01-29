from src.perspective import Calibration
from src.line import Line, LaneDetection
from src.filters import ColorFiltering, GradientFiltering, MorphologyFiltering
from src.annotate import AnnotateFrame
import numpy as np
import cv2
import glob


class LaneDetectionPipeline:
    def __init__(
        self,
        gradient_params=dict(
            sobel_kernel_size=3,
            sobel_threshold=(20, 100),
            magnitude_threshold=(50, 255),
            direction_threshold=(0.7, 1.3),
        ),
        color_params=dict(s_thresholds=(150, 255), r_thresholds=(200, 255)),
        morph_params=dict(kernel_size=(5, 5)),
        lane_detect_params=dict(
            num_windows=9,
            window_margin=50,
            min_pixels=50,
            fit_tolerance=100,
        ),
        annotate_params=dict(
            line_width=50,
            lane_colors=[(0, 0, 255), (255, 0, 0)],
            road_region_color=(0, 255, 0),
        ),
        line_object_params=dict(cache_size=15),
    ):
        self.gradient_params = gradient_params
        self.color_params = color_params
        self.morph_params = morph_params
        self.lane_detect_params = lane_detect_params
        self.annotate_params = annotate_params
        self.num_processed_frames = 0
        self.set_lane_objects(line_object_params["cache_size"])

    def set_lane_objects(self, cache_size):
        """ Define global line objects """
        self.left_lane = Line(cache_size=cache_size)
        self.right_lane = Line(cache_size=cache_size)

    def process_frame(self, img):
        """ pipeline that process and annotate a single frame """

        # apply color filters
        color = ColorFiltering(**self.color_params)
        color_filtered_img = color.apply_color_filter(img)

        # apply gradient filters
        gradient = GradientFiltering(**self.gradient_params)
        gradient_filtered_img = gradient.apply_gradient_filter(img)

        # combine gradient and color filter
        combined_filtered_image = np.logical_or(
            color_filtered_img, gradient_filtered_img
        )

        # apply morphology filter
        morph = MorphologyFiltering(**self.morph_params)
        moprhed_image = morph.apply_morphology_filter(combined_filtered_image)

        # calibrate the carmera
        perspective_transformer = Calibration()
        perspective_transformer.set_calibration()
        # undistort and convert image to bird's eye view
        birdeye_filtered_img = perspective_transformer.undistort_and_birdeye_transform(
            moprhed_image
        )
        birdeye_original_img = perspective_transformer.undistort_and_birdeye_transform(
            img
        )

        # detect lanes
        left_lane, right_lane, img_fit = LaneDetection(
            img=birdeye_filtered_img,
            left_lane=self.left_lane,
            right_lane=self.right_lane,
            **self.lane_detect_params,
        ).detect(self.num_processed_frames)
        # annotate frame
        annotate = AnnotateFrame(
            left_lane,
            right_lane,
            img_assets=[
                img,
                birdeye_original_img,
                moprhed_image,
                birdeye_filtered_img,
                img_fit,
            ],
            perspective_transformer=perspective_transformer,
            **self.annotate_params,
        )
        # produce the final blended frame
        res_frame = annotate.blend_frame()

        # increment processed frames by 1
        self.num_processed_frames += 1

        return res_frame


if __name__ == "__main__":
    # process all images in test folder
    for test_img in glob.glob("test_images/*.jpg"):
        img = cv2.imread(test_img)
        LaneDetectionPipeline(
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
        ).process_frame(img)
