
# Self Driving Car
Self Driving Car Projects from Udacity


### Project 1 - Finding Lanes

When we drive, we use our eyes to decide where to go. The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle. Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

In this project we will detect lane lines in images using Python and OpenCV. OpenCV means "Open-Source Computer Vision", which is a package that has many useful tools for analyzing images. We will build an algorithm that can detect lanes and annotate the location of the lanes as the car drives through. To be more specific, if we have a single scene of road image containing lanes like the following

<img src="images/solidWhiteRight.jpg" width="800" height="500">

We will build an algorithm that can annotate the same image like the following

<img src="images/laneLines_thirdPass.jpg" width="800" height="500">

We will then apply the alogorithm over some short video clips (series of images) and annotate the lanes as the car drives down the road, like the following.

<img src="images/solidYellowLeft_improved.gif" width="800" height="500">

### Project 2

With more advanced techiques we will be able to map out a clearer drivable region that is less affected by the noise on the road.

Continue with what was accomplished in project 1, a more robust software pipeline to identify the lane boundaries in a video will be created using Python and OpenCV. 

After processing by the pipeline, every scene will be annotated with lanes highlighted and drivable areas illumniated.


<img src="images/lane_detection.gif" width="800" height="500">


### Project 3

This project builds, trains and validates a model that can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). The trained model is subsequently tested on German traffic signs found on the web. The traffic sign recongnition pipeline is built with Tensorflow 2.4.1, containing the preprocessing steps and a subsequent deep convolutional neural network.

Sample results looks like the following

 <p align="left">
    <img src="examples/predict_test_images.png" width="600" height="130">
</p>






We will then apply the same pipeline to all scenes as the car drives down the road.
