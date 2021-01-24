from src.perspective import Calibration
from src.line import LaneDetection
from src.filters import ColorFiltering, GradientFiltering
from src.annotate import AnnotateFrame


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
            birdeye_img,
        ],  # to do: add image fit
        **annotate_params
    )
    # produce the final blended frame
    res_frame = annotate.blend_frame()

    return res_frame


if __name__ == "__main__":
    pass