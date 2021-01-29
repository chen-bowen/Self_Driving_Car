from src.line import LaneDetection
from src.perspective import Calibration
import cv2
import numpy as np
import matplotlib.pyplot as plt


class AnnotateFrame:
    """ Display road related information from detected lines and filtered image """

    def __init__(
        self,
        left_lane,
        right_lane,
        img_assets,
        perspective_transformer,
        line_width=100,
        lane_colors=[(0, 0, 255), (255, 0, 0)],
        road_region_color=(0, 255, 0),
    ):
        self.left_lane = left_lane  # left lane object
        self.right_lane = right_lane  # right lane object
        self.img_assets = img_assets  # intermeidate image assets
        self.perspective_transformer = (
            perspective_transformer  # perspective transform object
        )
        self.line_width = line_width  # lane line width on marker
        self.lane_colors = lane_colors  # lane line color
        self.road_region_color = road_region_color  # drivable region color
        self.get_img_assets()

    def get_img_assets(self):
        """
        Set image assets to their specific attributes
        """
        (
            self.image_original,
            self.img_warped,
            self.img_filtered,
            self.img_birdeye,
            self.img_fit,
        ) = self.img_assets

    def draw_lanes(self):
        """
        Draw lanes and green polygon back onto the image
        returns the processed image with lane lines drew over
        """
        # create base image
        lane_img = np.zeros_like(self.img_warped)

        # recast the x and y points into usable format for cv2.fillPoly()
        for i in range(len([self.left_lane, self.right_lane])):
            lane = [self.left_lane, self.right_lane][i]
            # get points with - half of width
            pts_l = np.array(
                [
                    np.transpose(
                        np.vstack(
                            [
                                lane.x_pix_values_continuous - self.line_width // 2,
                                lane.y_pix_values_continuous,
                            ]
                        )
                    )
                ]
            )
            # get points with + half of margin
            pts_r = np.array(
                [
                    np.flipud(
                        np.transpose(
                            np.vstack(
                                [
                                    lane.x_pix_values_continuous + self.line_width // 2,
                                    lane.y_pix_values_continuous,
                                ]
                            )
                        )
                    )
                ]
            )
            # stack the points to get line block
            pts = np.hstack((pts_l, pts_r))
            # draw line back onto base image
            cv2.fillPoly(lane_img, np.int32([pts]), self.lane_colors[i])

        # draw green polygon over the image
        pts_left = np.array(
            [
                np.transpose(
                    np.vstack(
                        [
                            self.left_lane.x_pix_values_continuous,
                            self.left_lane.y_pix_values_continuous,
                        ]
                    )
                )
            ]
        )
        pts_right = np.array(
            [
                np.flipud(
                    np.transpose(
                        np.vstack(
                            [
                                self.right_lane.x_pix_values_continuous,
                                self.right_lane.y_pix_values_continuous,
                            ]
                        )
                    )
                )
            ]
        )
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(lane_img, np.int_([pts]), self.road_region_color)

        # transform lane image in birds eye view back to normal
        lane_img_unwarp = self.perspective_transformer.birdeye_to_normal_transform(
            lane_img
        )

        # blend the lane image onto original image
        blended_img = cv2.addWeighted(self.image_original, 1, lane_img_unwarp, 0.5, 0)

        return blended_img

    def lane_info(self):
        """ returns lanes direction, curvature, deviation """
        # get average curvatures
        curvature = (
            self.left_lane.radius_of_curvature_in_meters
            + self.right_lane.radius_of_curvature_in_meters
        ) / 2
        # get lane width and center position
        lane_width = (
            self.right_lane.line_base_pos - self.left_lane.line_base_pos
        ) * LaneDetection.X_METERS_PER_PIX

        lane_center = self.img_warped.shape[1] / 2
        vehicle_center = (
            self.right_lane.line_base_pos + self.left_lane.line_base_pos
        ) / 2

        # calculate deviation
        deviation_meters = round(
            abs(vehicle_center - lane_center) * LaneDetection.X_METERS_PER_PIX, 2
        )
        deviation = (
            f"Left {deviation_meters} m"
            if vehicle_center > lane_center
            else f"Right {deviation_meters} m"
            if vehicle_center < lane_center
            else "Centered"
        )
        return lane_width, curvature, deviation

    def blend_frame(self):
        """ blend multiple image assets onto the orginal image scene """
        # draw lane lines over the warped image
        self.lane_fitted_img = self.draw_lanes()

        # define thumbnail ratio size
        h, w = self.lane_fitted_img.shape[:2]
        thumbnail_ratio = 0.2
        thumbnail_h, thumbnail_w = int(thumbnail_ratio * h), int(thumbnail_ratio * w)
        offset_x, offset_y = 20, 15

        # add a gray translucent rectangle as background
        translucent_background = self.lane_fitted_img.copy()
        translucent_background = cv2.rectangle(
            translucent_background,
            pt1=(0, 0),
            pt2=(w, thumbnail_h + 2 * offset_y),
            color=(0, 0, 0),
            thickness=cv2.FILLED,
        )
        self.lane_fitted_img = cv2.addWeighted(
            src1=translucent_background,
            alpha=0.2,
            src2=self.lane_fitted_img,
            beta=0.8,
            gamma=0,
        )

        # add thumbnails
        loc_idx = 1
        for img_asset in [self.img_filtered, self.img_birdeye, self.img_fit]:
            # thumbnail resize
            thumbnail = cv2.resize(img_asset, dsize=(thumbnail_w, thumbnail_h))
            # resize to 3-D if needed
            if len(img_asset.shape) < 3:
                thumbnail = np.dstack([thumbnail, thumbnail, thumbnail]) * 255

            # append to lane fitted image
            self.lane_fitted_img[
                offset_y : (thumbnail_h + offset_y),
                (loc_idx * (offset_x + thumbnail_w) - thumbnail_w) : (
                    loc_idx * (offset_x + thumbnail_w)
                ),
                :,
            ] = thumbnail

            loc_idx += 1

        # add text regarding lane information
        font = cv2.FONT_HERSHEY_SIMPLEX
        lane_width, curvature, deviation = self.lane_info()
        # lane_width
        cv2.putText(
            self.lane_fitted_img,
            "Lane Width: {:.02f}m".format(lane_width),
            (860, 60),
            font,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # curvature
        cv2.putText(
            self.lane_fitted_img,
            "Curvature Radius: {:.02f}m".format(curvature),
            (860, 130),
            font,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        # deviation
        cv2.putText(
            self.lane_fitted_img,
            "Deviation: {}".format(deviation),
            (860, 200),
            font,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return self.lane_fitted_img
