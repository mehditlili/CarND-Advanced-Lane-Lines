##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/original_pattern.jpg "Original"
[image2]: ./output_images/undistorted_pattern.jpg "Undistorted"
[image3]: ./output_images/original_image.jpg 
[image4]: ./output_images/filtered_image.jpg  "Warp Example"
[image5]: ./output_images/warped_color.jpg  "Warp Example"
[image6]: ./output_images/warped_cropped.jpg "Fit Visual"
[image7]: ./output_images/roi.jpg 
[image8]: ./output_images/histogram.jpg 
[image9]: ./output_images/fitting.jpg 
[image11]: ./output_images/test0_result.jpg 
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I created a class called Camera which has the main task of calibrating the camera and offers an
undistort function to remove radial and tangential radiation from images.


I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. 
Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each 
calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy
of it every time I successfully detect all chessboard corners in a test image.  
`imgpoints` will be appended with the (x, y) pixel position of each of the corners 
in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using 
the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test 
image using the `cv2.undistort()` function and obtained this result: 

Original Pattern

![alt text][image1]

Undistorted Pattern

![alt text][image2]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image3]
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image.
All filtering functions were implemented int he class FilterTools.

A combination of those functions was then used in order to generate an optimal filtered image resulting
of the combination of filtered image using different methods, intensity based and color based.
Here's an example of my output for this step.

![alt text][image4]

This filtering is done on the original image, before the undistorting.

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, 
The `warp()` function takes as inputs a binary filtered image (`filtered_img`), 
as well as color image (`original_image`).
In my code the perspective warp matrix is defined globally as well as the destination shape
that I chose to be 800x800.

I chose the hardcode the source and destination points 
in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the region of interest
in the original image, then warping that image

Here is the image with its respective region of interest (source points) 

![alt text][image7]

When warped one can see that the yellow ROI edges appear indeed parallel so the perspective
transform is valid.

![alt text][image5]

And the result when applying the same warping to the binary image shown above:

![alt text][image6]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Once the warped binary image is available, I compute a histogram of intensities by summing
up all the rows. Expecting lane lines to appear vertical in the warped image, the histogram is 
expected to have to clearly identifiable maximas. One towards the left of the image and one towards
the right of the image.

Here Is an example histogram corresponding to the image above:


![alt text][image8]

Running the function 'search_for_lines' on this patch returns 2 degrees polynomial fitting
to the vertical white blobs found around the histogram maximas.
The following images shows the white blobs detected at the histogram maxima (red and blue)

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Here is an example of my result on a test image:

![alt text][image11]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://youtu.be/IfYqZVhLC34)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

