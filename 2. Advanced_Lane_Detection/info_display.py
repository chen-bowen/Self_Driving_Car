from line import Line, LaneDetection
import cv2
import numpy as np
import matplotlib.pyplot as plt


class annotateFrame:
    """ Display road related information from detected lines and filtered image """

    def __init__(
        self,
        left_lane,
        right_lane,
        img_assets,
        line_width=50,
        lane_color=(255, 0, 0),
        road_region_color=(0, 255, 0),
    ):
        self.left_lane = left_lane
        self.right_lane = right_lane
        self.img_assets = img_assets
        self.line_width = line_width
        self.lane_color = lane_color
        self.road_region_color = road_region_color

    def get_img_assets(self):
        """
        Set image assets to their specific attributes
        """
        self.img_filtered, self.img_birdeye, self.img_warped = self.img_assets

    def draw_lanes(self):
        """
        Draw lanes and green polygon back onto the image
        returns the processed image with lane lines drew over
        """
        # create base image
        lane_img = np.zeros_like(self.img_warped)

        # recast the x and y points into usable format for cv2.fillPoly()
        for lane in [self.left_lane, self.right_lane]:
            # get points with - half of width
            pts_l = np.array(
                [
                    np.transpose(
                        np.vstack(
                            [
                                lane.x_pix_values - self.line_width // 2,
                                lane.y_pix_values,
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
                                    lane.x_pix_values + self.line_width // 2,
                                    lane.y_pix_values,
                                ]
                            )
                        )
                    )
                ]
            )
            # stack the points to get line block
            pts = np.hstack((pts_l, pts_r))
            # draw line back onto base image
            cv2.fillPoly(lane_img, np.int32([pts]), self.lane_color)

        # draw green polygon over the image
        pts_left = np.array(
            [
                np.transpose(
                    np.vstack(
                        [self.left_lane.x_pix_values, self.left_lane.y_pix_values]
                    )
                )
            ]
        )
        pts_right = np.array(
            [
                np.flipud(
                    np.transpose(
                        np.vstack(
                            [self.right_lane.x_pix_values, self.right_lane.y_pix_values]
                        )
                    )
                )
            ]
        )
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(lane_img, np.int_([pts]), self.road_region_color)

        # blend the lane image onto original image
        return cv2.addWeighted(self.img_warped, 1, lane_img, 0.3, 0)

    def lane_info(self):
        """ returns lanes direction, curvature, deviation """
        # get average curvatures
        curvature = (
            self.left_lane.radius_of_curvature_in_meters
            + self.right_lane.radius_of_curvature_in_meters
        ) / 2
        # get lane width and center position
        lane_width = self.right_lane.line_base_pos - self.left_lane.line_base_pos
        lane_center = self.processed_img.shape[1] / 2
        vehicle_center = (
            self.right_lane.line_base_pos + self.left_lane.line_base_pos
        ) / 2

        # calculate deviation
        deviation_meters = round(abs(vehicle_center - lane_center), 2)
        deviation = (
            f"Left {deviation_meters} m"
            if vehicle_center > lane_center
            else f"Right {deviation_meters} m"
            if vehicle_center < lane_center
            else "Centered"
        )
        return curvature, deviation

    def blend_frame(self, )