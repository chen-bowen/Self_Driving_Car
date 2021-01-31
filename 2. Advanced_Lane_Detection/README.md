## Advanced Lane Finding

<p align="center">
  <img src="examples/lane_detection.gif" width="800" height="500">
</p>

##
### Project Overview

Continue with what was accomplished in project 1, a more robust software pipeline to identify the lane boundaries in a video will be created using Python and OpenCV. Starting from an image scene like below

<p align="center">
  <img src="output_images/original_img.jpg" width="800" height="500">
</p>


After processing by the pipeline the same image will be annotated with lanes highlighted and drivable areas illumniated.


<p align="center">
  <img src="output_images/annotated_scene.jpg" width="800" height="500">
</p>


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

Using the HSV space filter, the noise of the image will be remnoved with the lanes singled out. By separating the image's R, G, B channel, it is evident that R channel (red) has some solid abilities to pick up the left lane

<p align="center">
  <img src="examples/r_filter.png" width="800" height="500">
</p>


To strengthen that effect, the image in HSV color space's s channel (saturation) also does a great job for picking out the left lane. 

<p align="center">
  <img src="examples/s_filter.png" width="800" height="500">
</p>


Using an or logic gate could allow 2 filters to cover up each other's mistakes. The combined filter produced a binary filter that is specialized in finding the left lane. The combined filtering effect is shown below

<p align="center">
  <img src="output_images/color_filter.jpg" width="800" height="500">
</p>

The relevant module for building this filters is in `filters.py`

### Gradient Filtering

By restricting the gradient of the image of a single direction, lane objects could be emphasized in the resulting binary image. For example, restricting gradient in x and y direction will result in image binaries like the following


Gradient X          |  Gradient Y
:-------------------------:|:-------------------------:
<img src="examples/Sobel_X.png" width="500" height="300"> | <img src="examples/Sobel_Y.png" width="500" height="300">

Gradient on y direction specializes in detecting lines in horizontal direction, since lane lines are mostly vertical in the scenes, only gradient x filter is used.


Restricing magnitudes of the image's gradient could also be helpful in emphasizing lane lines. Applying a magnitude filter will result in a binary image like the following

<p align="center">
    <img src="examples/magnitude.png" width="800" height="500">
</p>

magnitude filters produced a smoothed binary image that could help connecting the small segments that could help detecting lane lines. Combining the the gradient filters produced a binary image for the oringinal scene like the following

<p align="center">
    <img src="output_images/gradient_filter.jpg" width="800" height="500">
</p>

The color filters and gradient filters will be combined with a logical or gate to pick up as much lane information as possible. The relevant module for building this gradient filter is in `filters.py`

### Camera Calibration and Image Undistortion

To properly undistort the images and make the measurements of the road robust, the images taken by a certain camera needs to be properly calibrated. CV2 has provided a convient way of calibrating the camera using chessboard images. The images for camera calibration are stored in the folder called `camera_cal`. By sequentially feeding images into cv2's `findChessboardCorners` method, the object and image points can be obtained and used for the cv2's `calibrateCamera` method. For example, after undistortion, the effect on one of the chessboards is the following

Chessboard 1 Original       |  Chessboard 1 Undistorted 
:-------------------------:|:-------------------------:
<img src="camera_cal/calibration1.jpg" width="500" height="300"> | <img src="output_images/camera_cal_1_undistort.jpg" width="500" height="300">

Once the camera is properly calibrated, the distortion matrix and coefficients could be used to calibrate the real road scene.

Road Scene Original       |  Road Scene Undistorted 
:-------------------------:|:-------------------------:
<img src="output_images/original_img.jpg" width="500" height="300"> | <img src="output_images/undistort_image.jpg" width="500" height="300">

### Birdeye Transform

As objects appear larger when they are closer, directly taking measurements of the road using the images taken by the car's front camera will result in erroneuous values. The best way to normalize the effect is looking at the scene from a birdeye view. Using the birdeye view also served as a road-only filter as it crops out the extra irrelevant objects around the scene that were picked up by the previous color and gradient filters. CV2 also provided a convenient method to accomplish such goal - `getPerspectiveTransform`. The method requires source and destination points to be properly defined. Same as the previous project, the region of interest is the polygon right in front of the car, which will help define the src points for the perspective transformation. 

<p align="center">
    <img src="examples/canny.jpg" width="800" height="500">
</p>
 

To be specific, if the height and width of the image is h and w respectively, the source points (defined empirically) are 

```python

src = np.float32(
    [
        [w, h - 10],  # bottom right
        [0, h - 10],  # bottom left
        [546, 460],  # top left, lane top left corner from polygon
        [732, 460],  # top right, lane top right corner from polygon
    ]
)

```

The destination points will simply be the 4 corners of an image that is the same as the input image.

After the perspective transform, the image will become the following 

<p align="center">
    <img src="output_images/perspective_transform_image.jpg" width="800" height="500">
</p>
 

