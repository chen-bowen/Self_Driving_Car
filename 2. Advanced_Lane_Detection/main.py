from src.perspective import Calibration
from src.line import Line, LaneDetection
from src.filters import ColorFiltering, GradientFiltering
from src.annotate import AnnotateFrame
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
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
        save_output=False,
    ):
        self.gradient_params = gradient_params
        self.color_params = color_params
        self.lane_detect_params = lane_detect_params
        self.annotate_params = annotate_params
        self.num_processed_frames = 0
        self.save_output = save_output
        self.set_lane_objects(line_object_params["cache_size"])
        self.set_calibration()

    def set_lane_objects(self, cache_size):
        """ Define global line objects """
        self.left_lane = Line(cache_size=cache_size)
        self.right_lane = Line(cache_size=cache_size)

    def set_calibration(self):
        """ Set the calibration camera once"""
        self.perspective_transformer = Calibration()
        self.perspective_transformer.set_calibration()

    def process_frame(self, img):
        """ pipeline that process and annotate a single frame """

        # apply color filters
        color = ColorFiltering(**self.color_params)
        color_filtered_img = color.apply_color_filter(img, self.save_output)

        # apply gradient filters
        gradient = GradientFiltering(**self.gradient_params)
        gradient_filtered_img = gradient.apply_gradient_filter(img, self.save_output)

        # combine gradient and color filter
        combined_filtered_image = np.logical_or(
            color_filtered_img, gradient_filtered_img
        ).astype(np.uint8)

        # undistort and convert image to bird's eye view
        birdeye_filtered_img = (
            self.perspective_transformer.undistort_and_birdeye_transform(
                combined_filtered_image, self.save_output
            )
        )
        birdeye_original_img = (
            self.perspective_transformer.undistort_and_birdeye_transform(
                img, self.save_output
            )
        )
        # update lane caches
        self.left_lane.update_cache()
        self.right_lane.update_cache()

        # detect lanes
        left_lane, right_lane, img_fit = LaneDetection(
            img=birdeye_filtered_img,
            left_lane=self.left_lane,
            right_lane=self.right_lane,
            **self.lane_detect_params,
        ).detect(self.num_processed_frames, self.save_output)
        # annotate frame
        annotate = AnnotateFrame(
            left_lane,
            right_lane,
            img_assets=[
                img,
                birdeye_original_img,
                combined_filtered_image,
                birdeye_filtered_img,
                img_fit,
            ],
            perspective_transformer=self.perspective_transformer,
            **self.annotate_params,
        )
        # produce the final blended frame
        res_frame = annotate.blend_frame(self.save_output)

        # increment processed frames by 1
        self.num_processed_frames += 1

        return res_frame


if __name__ == "__main__":
    from moviepy.editor import VideoFileClip

    # process all images in test folder
    test_img = glob.glob("test_images/*.jpg")[-1]
    img = mpimg.imread(test_img)
    plt.imsave("output_images/original_img.jpg", img)
    lane_detect = LaneDetectionPipeline(
        gradient_params=dict(
            sobel_kernel_size=3,
            sobel_threshold=(70, 150),
            magnitude_threshold=(70, 255),
            direction_threshold=(0.7, 1.3),
        ),
        color_params=dict(s_thresholds=(150, 255), r_thresholds=(200, 255)),
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
        save_output=True,
    )
    lane_detect.process_frame(img)

    # produce an annotated video
    # lane_detect.save_output = False
    # vid_output = "test_videos_output/project_video.mp4"
    # clip1 = VideoFileClip("test_videos/project_video.mp4")
    # project_clip = clip1.fl_image(lane_detect.process_frame)
    # project_clip.write_videofile(vid_output, audio=False)