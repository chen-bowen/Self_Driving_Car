import cv2
import numpy as np
import os
import pickle


class Calibration:
    """ camera calibration, image undistort and perspective transformation on the image """

    def __init__(self, img_dir="camera_cal"):
        self.set_calibration_img_dir(img_dir)
        self.set_corners_paramters()

    def set_corners_paramters(self):
        """set the calibration x and y as part of the properties"""
        self.nx = 9
        self.ny = 6
        self.corners_offset = 100

    def set_calibration_img_dir(self, img_dir):
        """set the directory of all the calibration images"""
        self.cal_img_dir = img_dir

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
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, self.image_size, None, None
        )
        # Save the camera calibration result for later use
        dist_pickle = {}
        dist_pickle["mtx"] = mtx
        dist_pickle["dist"] = dist
        pickle.dump(dist_pickle, open("calibration_model/wide_dist_pickle.p", "wb"))

    def undistort_image(self, img):
        """ undistort the input image using calibrated camera """
        # set camera calibration
        self.set_calibration()
        # undistort image
        undistored_image = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

        return undistored_image

    def perspective_transform_image(self, undistort_img):
        """ Perform perspective transform on the input undistorted image """
        # Convert undistorted image to grayscale
        gray = cv2.cvtColor(undistort_img, cv2.COLOR_BGR2GRAY)
        # Search for corners in the grayscaled image
        found, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if found == True:
            # define source points for the perspective transform
            src = np.float32([corners[0], corners[nx - 1], corners[-1], corners[-nx]])
            # define the destination points
            dst = np.float32(
                [
                    [offset, offset],
                    [self.image_size[0] - offset, offset],
                    [self.image_size[0] - offset, self.image_size[1] - offset],
                    [offset, self.image_size[1] - offset],
                ]
            )
            # Given src and dst points, calculate the perspective transform matrix
            M = cv2.getPerspectiveTransform(src, dst)
            # Warp the image using OpenCV warpPerspective
            warped = cv2.warpPerspective(undistort_img, M, self.image_size)

        return warped, M

    def calibrate(self, img):
        """ return the undistort and perspective transform image """
        # set image size
        self.image_size = (img.shape[1], img.shape[0])
        # undistort image
        undistored_img = self.undistort_image(img)
        # perspective transform the image
        calibrated_image = self.perspective_transform(undistored_img)

        return calibrated_image
