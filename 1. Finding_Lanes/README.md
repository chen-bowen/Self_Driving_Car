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
