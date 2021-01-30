import cv2
import numpy as np
import os
import pickle
import glob
import matplotlib.pyplot as plt


class Calibration:
    """ camera calibration, image undistort and perspective transformation on the image """

    def __init__(self, img_dir="../camera_cal"):
        self.set_calibration_img_dir(img_dir)
        self.set_corners_paramters()

    def set_corners_paramters(self):
        """set the calibration x and y as part of the properties"""
        self.nx = 9
        self.ny = 6
        self.corners_offset = 100

    def set_calibration_img_dir(self, img_dir):
        """set the directory of all the calibration images and destination of calibration model"""
        current_file_path = os.path.dirname(os.path.realpath(__file__))
        self.cal_img_dir = current_file_path + "/" + img_dir
        self.cal_model_dir = current_file_path + "/calibration_model/wide_dist_pickle.p"

    def find_chess_board_corners(self):
        """
        Find the all the chess board corners of the images under image calibration dir
        returns a list of corners for each image
        """
        # initialize the object points list and corners points list
        objpoints = []
        imgpoints = []
        # build object points
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        # get all the images in the calibration image directory
        cal_images = glob.glob(f"{self.cal_img_dir}/calibration*.jpg")

        for img_name in cal_images:
            # read the image and convert image to gray scale
            img = cv2.imread(img_name)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.gray_shape = gray.shape[::-1]
            # Find the chessboard corners
            found, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)
            # If found, append the object and image pages
            if found == True:
                objpoints.append(objp)
                imgpoints.append(corners)

        return imgpoints, objpoints

    def set_calibration(self):
        """use calibrated point to distort the input image """
        # use archived calibration model if exists
        if os.path.exists("calibration_model/wide_dist_pickle.p"):
            dist_pickle = pickle.load(open("wide_dist_pickle.p", "rb"))
            self.dist = dist_pickle["dist"]
            self.mtx = dist_pickle["mtx"]
            return
        # get image and object points
        imgpoints, objpoints = self.find_chess_board_corners()
        # return distortion transfomration matrix and distances
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, self.gray_shape, None, None
        )
        # Save the camera calibration result for later use
        dist_pickle = {}
        dist_pickle["mtx"] = self.mtx
        dist_pickle["dist"] = self.dist

        pickle.dump(dist_pickle, open(self.cal_model_dir, "wb"))

    def undistort_image(self, img, save_ouput=False):
        """ undistort the input image using calibrated camera """
        # undistort image
        undistored_image = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

        if save_ouput:
            # save output to output_images
            plt.imsave(
                "output_images/undistort_image.jpg", undistored_image, cmap="gray"
            )

        return undistored_image

    def perspective_transform_image(self, undistort_img, save_ouput=False):
        """ Perform perspective transform on the input undistorted 2-D image """

        # define source points for the perspective transform
        h, w = undistort_img.shape[:2]

        src = np.float32(
            [
                [w, h - 10],  # bottom right
                [0, h - 10],  # bottom left
                [546, 460],  # top left, lane top left corner from polygon
                [732, 460],  # top right, lane top right corner from polygon
            ]
        )
        dst = np.float32(
            [
                [w, h],  # bottom right
                [0, h],  # bottom left
                [0, 0],  # top left
                [w, 0],  # top right
            ]
        )
        # Given src and dst points, calculate the perspective transform matrix and its inverse
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inverse = cv2.getPerspectiveTransform(dst, src)
        # Warp the image using OpenCV warpPerspective
        warped = cv2.warpPerspective(undistort_img, self.M, (w, h))

        if save_ouput:
            # save output image to output_images
            plt.imsave(
                "output_images/perspective_transform_image.jpg", warped, cmap="gray"
            )

        # return warped image only found the object
        return warped

    def undistort_and_birdeye_transform(self, img, save_ouput=False):
        """ return the undistort and perspective transform image """
        # undistort image
        undistored_img = self.undistort_image(img, save_ouput)
        # perspective transform the image
        calibrated_image = self.perspective_transform_image(undistored_img, save_ouput)

        return calibrated_image

    def birdeye_to_normal_transform(self, birdeye_img):
        """ Convert birdeye image back to normal view """
        # get height and width of the bird's eye view
        height, width, _ = birdeye_img.shape
        # convert back to normal view
        img = cv2.warpPerspective(birdeye_img, self.M_inverse, (width, height))

        return img
