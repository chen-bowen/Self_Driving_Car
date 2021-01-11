# **Finding Lane Lines on the Road** 


### Project Overview

When we drive, we use our eyes to decide where to go. The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle. Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

In this project we will detect lane lines in images using Python and OpenCV. OpenCV means "Open-Source Computer Vision", which is a package that has many useful tools for analyzing images. We will build an algorithm that can detect lanes and annotate the location of the lanes as the car drives through. To be more specific, if we have a single scene of road image containing lanes like the following

![image1](test_images/solidWhiteRight.jpg)

We will build an algorithm that can annotate the same image like the following

![image2](examples/laneLines_thirdPass.jpg)

We will then apply the alogorithm over some short video clips (series of images) and annotate the lanes as the car drives down the road.  


### Lane Detection Pipeline

The algorithm follows a series of steps that eventually has the ablity to annotate the lane lines over the image. These steps are packaged inside a pipeline with the following steps,

1. Color Filtering
2. Gaussian Blur
3. Canny Edge Detection
4. Region of Interest Selection
5. Hough Lines Overlay
6. Image Annotation

**Color Filtering**

As most people already recognized, most lanes we see in real life are either white or yellow. To properly detect lanes, we can build a mechanism that only recogonize white or yellow colors, which will block out all the objects that are irrelevant. To begin with, the image will be converted into the HSV space to reduce the dimensionality to 2. Then the color filter could be built by restricting the values pass through images' 3 color channels - only values within a certain range will be allowed, with all other values set to 0. 

To get only white colors, we will set a lower bound of values on the image 0, 150, 0 on image's RGB channels, respectively. The upper bound of this white filter will be 255, 255, 255 (effectively no upper bound since it's reaching the color channels' maximum values).

By the same way, yellow colors bounds will be [40, 0, 100], [50, 255, 255]. The result image is the following

![image3](examples/color_filter.jpg)

Most irrelevant objects (cars, background sceneries) are immediately blocked out with this filter.

**Gaussian Blur**

Lanes normally are built to have high contrast with the surrounding backgrounds, which could be used as features from the edge detection algorithms. However, they are also very thin, often making the gradient very big. To solve this issue, gaussian blur could be applied, which will make the contrast transition smoother. After gaussian blur, the resulting image looks like the following.

![image4](examples/gaussian_blur.jpg)

Compared with the previous image, the resulting image after this step is slightly blurred than the previous one.

**Canny Edge Detection**

Canny's algorithm is specifically used to detect sharp changes in color around the image (gradient). Lanes presents a sharp change in color compared to its surroundings, therefore will be detected by Canny edge detection. After this operation, the resulting image becomes this

![image5](examples/canny_edge_detect.jpg)

**Region of Interest**

Lanes are clearly mapped out after this process, so are some other irrelevant objects however. This is due to those irrelevant objects being white or yellow. The way to solve this issue is to persist only the region that contains our lane lines. In general, we are interested in trapzoid region which represents the view from the car's front camera.

![image6](examples/canny_edge_detect_region.jpg)

Using cv2's fitpoly function, we are about to define a mask over the image using the appropriate vertices. The mask will only allow values in the region of interest to pass through, while setting values of all positions outside that region of interest to 0. We will define the vertices as a ratio of image size. For example, the top 2 points' x values are approximately 0.6 x image height, while the y values are hovers around 0.5 of image width (0.48 and 0.55 respectively). With the mask built, we can get a clean outline of the lanes, as shown in the following.

![image7](examples/region_masked.jpg)



### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I .... 

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
