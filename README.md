# Self Driving Car
Self Driving Car Projects from Udacity


### Project 1 - Finding Lanes

When we drive, we use our eyes to decide where to go. The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle. Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

In this project we will detect lane lines in images using Python and OpenCV. OpenCV means "Open-Source Computer Vision", which is a package that has many useful tools for analyzing images. We will build an algorithm that can detect lanes and annotate the location of the lanes as the car drives through. To be more specific, if we have a single scene of road image containing lanes like the following

<img src="test_images/solidWhiteRight.jpg" width="800" height="500">

We will build an algorithm that can annotate the same image like the following

<img src="examples/laneLines_thirdPass.jpg" width="800" height="500">

We will then apply the alogorithm over some short video clips (series of images) and annotate the lanes as the car drives down the road, like the following.

<img src="examples/solidYellowLeft_improved.gif" width="800" height="500">