## Advanced Lane Finding

<img src="examples/lane_detection.gif" width="800" height="500">

Continue with what was accomplished in project 1, a more robust software pipeline to identify the lane boundaries in a video will be created using Python and OpenCV. Starting from an image scene like below

<img src="test_images/test3.jpg" width="800" height="500">

After processing by the pipeline the same image will be annotated with lanes highlighted and drivable areas illumniated.

<img src="test_images/full_transform_test3.jpg" width="800" height="500">

We will then apply the same pipeline to all scenes as the car drives down the road.


The  steps of the pipeline are the following:

* Use color transforms, gradients, etc., to create a thresholded binary image
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images
* Apply a distortion correction to raw images
* Apply a perspective transform to rectify binary image ("birds-eye view")
* Detect lane pixels and fit to find the lane boundary
* Determine the curvature of the lane and vehicle position with respect to center
* Warp the detected lane boundaries back onto the original image
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position

The whole pipeline was developed in object oriented programming of 6 different classes. `GradientFiltering` and `ColorFiltering` contains the binary filters that would help the pipeline segment out the potential location of lane lines. `Calibration` contains methods to calibrate the camera, undistort and warp images into bird eye views. `Line` defines a line objects which could be used to save some past information about the line, such as slopes and curvature. Finally `LaneDectionPipeline` builds the software pipeline that transform and annotate each frame of the video.

### Color Filtering

Using the HSV space filter, the noise of the image 

The images for camera calibration are stored in the folder called `camera_cal`.  

The images in `test_images` are for testing your pipeline on single frames.  If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  
 

