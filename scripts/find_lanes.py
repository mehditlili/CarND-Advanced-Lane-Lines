import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import moviepy
from moviepy.editor import VideoFileClip
import copy

'''
Advanced Lane Finding Project

The goals / steps of this project are the following:

Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
Apply a distortion correction to raw images.
Use color transforms, gradients, etc., to create a thresholded binary image.
Apply a perspective transform to rectify binary image (“birds-eye view”).
Detect lane pixels and fit to find the lane boundary.
Determine the curvature of the lane and vehicle position with respect to center.
Warp the detected lane boundaries back onto the original image.
Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
'''

'''
Part1:
read all chessboard images, detect the patterns and compute the camera calibration matrix
'''

warped_shape = None
transform = None
cam = None
left_fit = None
roi = None
right_fit = None
last_one_good = False

class Camera(object):

    def __init__(self):
        self.debug = False  # display or not the loaded picture
        self.intrinsic = None  # camera matrix
        self.distortion = None  # camera distortion
        self.img_shape = None
        # Arrays to store object points and image points from all the images.
        self.obj_points = []  # 3d points in real world space
        self.img_points = []  # 2d points in image plane.

    def read_chessboards(self, img_path):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        images = glob.glob(img_path)
        # Step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if self.img_shape is None:
                self.img_shape = gray.shape[::-1]
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
            # If found, add object points, image points
            if ret == True:
                self.obj_points.append(objp)
                self.img_points.append(corners)
                if self.debug:
                    # Draw and display the corners
                    img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
                    cv2.imshow('img', img)
                    cv2.waitKey(500)

    def calibrate_camera(self, img_path):
        # Compute camera calibration using the chessboards
        ret, self.intrinsic, self.distortion, rvecs, tvecs = cv2.calibrateCamera(self.obj_points,
                                                                                 self.img_points,
                                                                                 self.img_shape,
                                                                                 None,
                                                                                 None)
        images = glob.glob(img_path)
        for fname in images:
            img = cv2.imread(fname)
            dst = self.undistort(img)
            if self.debug:
                cv2.imshow('undistorted', dst)
                cv2.waitKey(500)

    def undistort(self, img):
        if self.intrinsic is not None and self.distortion is not None:
            return cv2.undistort(img, self.intrinsic, self.distortion)
        else:
            print("Camera was not calibrated yet")
            exit(1)


class FilterTools(object):
    @staticmethod
    def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
        thresh_min = mag_thresh[0]
        thresh_max = mag_thresh[1]
        # Apply the following steps to img
        # 1) Convert to grayscale
        gray = img
        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # 3) Calculate the magnitude
        mag_sobel = np.sqrt(np.power(sobelx, 2) + np.power(sobely, 2))
        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255 * mag_sobel / np.max(mag_sobel))
        # 5) Create a mask of 1's where the scaled gradient magnitude
        # is > thresh_min and < thresh_max
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        # 6) Return this mask as your binary_output image
        return binary_output
    @staticmethod
    def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
        thresh_min = thresh[0]
        thresh_max = thresh[1]
        # Apply the following steps to img
        # 1) Convert to grayscale
        gray = img
        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # 3) Take the absolute value of the x and y gradients
        abs_sobelx = np.abs(sobelx)
        abs_sobely = np.abs(sobely)
        # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
        dir_sobel = np.arctan2(abs_sobely, abs_sobelx)
        # 5) Create a binary mask where direction thresholds are met
        binary_output = np.zeros_like(dir_sobel)
        binary_output[(dir_sobel >= thresh_min) & (dir_sobel <= thresh_max)] = 1
        # 6) Return this mask as your binary_output image
        return binary_output

    @staticmethod
    def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        # Apply the following steps to img
        # 1) Convert to grayscale
        thresh_min = thresh[0]
        thresh_max = thresh[1]
        gray = img
        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        else:
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # 3) Take the absolute value of the derivative or gradient
        abs_sobel = np.absolute(sobel)
        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        # 5) Create a mask of 1's where the scaled gradient magnitude
        # is > thresh_min and < thresh_max
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        # 6) Return this mask as your binary_output image
        return binary_output

    @staticmethod
    def intensity_thresh(img, thresh=(0, 255)):
        thresh_min = thresh[0]
        thresh_max = thresh[1]
        binary_output = np.zeros_like(img)
        binary_output[(img >= thresh_min) & (img <= thresh_max)] = 1
        # 6) Return this mask as your binary_output image
        return binary_output

    @staticmethod
    def hls_select(img, thresh=(0, 255)):
        s_channel = img
        binary_output = np.zeros_like(s_channel)
        binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
        # 3) Return a binary image of threshold result
        return binary_output

    @staticmethod
    def hsv_select(img):
        # Extact yellow and white colors from image and store them in the variable combined
        hsv_image = img
        # define range of blue and white color in HSV (Im my case due to some codec problem yellow appears as blue in cv2
        lower_blue = np.array([0, 0, 0])
        upper_blue = np.array([30, 225, 255])
        lower_white = np.array([90, 5, 210])
        upper_white = np.array([160, 30, 255])
        # cv2.imshow("hue", hsv_image[:, :, 0])
        # cv2.imshow("saturation", hsv_image[:, :, 1])
        # cv2.imshow("value", hsv_image[:, :, 2])
        # Additional filtering to extract only yellow and white colors from image
        mask_yellow = cv2.inRange(hsv_image, lower_blue, upper_blue)
        mask_white = cv2.inRange(hsv_image, lower_white, upper_white)
        # combine yellow and white image
        combined = cv2.bitwise_or(mask_white, mask_yellow)
        # Dilate and erode, more iterations for erode to remove single points
        kernel = np.ones((3, 3), np.uint8)
        combined = cv2.dilate(combined, kernel, iterations=3)
        combined = cv2.erode(combined, kernel, iterations=2)
        return combined

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None


def search_for_lines(binary_warped, out_img):
    histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit, left_lane_inds, right_lane_inds

def open(image):
    # Dilate and erode, more iterations for erode to remove single points
    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.dilate(image, kernel, iterations=3)
    combined = cv2.erode(combined, kernel, iterations=2)
    return combined

def predict_lines(binary_warped, left_fit, right_fit):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
    nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = (
    (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
    nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit, left_lane_inds, right_lane_inds
    # Generate x and y values for plotting
    #ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    #left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    #right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

def good_lines(left_fit, right_fit):
    print("left fit: %s" % left_fit)
    print("right fit: %s" % right_fit)
    return True


def find_lines(binary_warped, image, Minv):
    global left_fit
    global right_fit
    global last_one_good
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    if left_fit is None or right_fit is None or not last_one_good:
        new_left_fit, new_right_fit, left_lane_inds, right_lane_inds = search_for_lines(binary_warped, out_img)
    else:
        new_left_fit, new_right_fit, left_lane_inds, right_lane_inds = predict_lines(binary_warped, left_fit, right_fit)
    if good_lines(new_left_fit, new_right_fit):
        left_fit, right_fit = new_left_fit, new_right_fit
    else:
        last_one_good = False


    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    # plt.imshow(out_img)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 800)
    # plt.ylim(800, 0)
    # plt.show()
    cv2.imshow("processing", out_img)
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    print(image.shape)
    print(newwarp.shape)
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    cv2.imshow("result", result)
    # cv2.waitKey(5)
    return result


def filter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(gray, gray)
    s = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 2]
    #cv2.equalizeHist(s, s)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv_thresh = FilterTools.hsv_select(hsv)
    cv2.imshow("hsv", hsv_thresh*255)
    print(hsv_thresh)
    s_sobel_x = FilterTools.abs_sobel_thresh(s, 'x', 7, (20, 100))
    s_sobel_y = FilterTools.abs_sobel_thresh(s, 'y', 7, (10, 100))
    sobel_x = FilterTools.abs_sobel_thresh(gray, 'x', 7, (30, 100))
    sobel_y = FilterTools.abs_sobel_thresh(gray, 'y', 7, (30, 100))
    mag_s_sobel = FilterTools.mag_thresh(s, 3, (10, 20))
    mag_s_sobel = open(mag_s_sobel)
    cv2.imshow("mag", mag_s_sobel*255)
    mag_sobel = FilterTools.mag_thresh(gray, 3, (10, 20))
    mag_sobel = open(mag_s_sobel)
    # cv2.imshow("maggray", mag_sobel*255)
    # cv2.imshow("sobel_x", sobel_x*255)
    # cv2.imshow("sobel_y", sobel_y*255)
    # cv2.imshow("s_sobel_x", s_sobel_x*255)
    # cv2.imshow("s_sobel_y", s_sobel_y*255)
    cv2.imshow("gray", gray)
    s_thresh = FilterTools.hls_select(s, (70, 255))
    dir_sobel = FilterTools.dir_threshold(gray, 3, (0.75, 0.8))
    s_dir_sobel = FilterTools.dir_threshold(s, 3, (0.75, 0.8))

    intensity = FilterTools.intensity_thresh(gray, (40, 255))
    combined = np.zeros_like(s)
    combined[intensity == 1 & (((s_sobel_x == 1) & (s_sobel_y == 1)) |
             ((sobel_x == 1) & (sobel_y == 1)) |
             (s_thresh == 1))] = 1
    cv2.imshow("combined", combined*255)
    #cv2.waitKey(5)
    return combined


def warp(filtered_img, original_img):
    global cam
    global transform
    global warped_shape
    undistorted = cam.undistort(filtered_img)
    undistorted_color = cam.undistort(original_img)
    warped = cv2.warpPerspective(undistorted, transform, tuple(warped_shape))
    cv2.imshow("warped", warped*255)
    return warped, undistorted_color


def display_image(img):
    cv2.imshow("test", img)
    cv2.waitKey(100)
    return img

def draw_roi(img, roi):
    pts = roi.reshape((-1, 1, 2))
    print(pts)
    cv2.polylines(img, [pts.astype(np.int32)], True, (0, 255, 255))
    return img

def crop_edges(img, margin):
    img[:, :margin] = 0
    img[:margin, :] = 0
    img[-margin:, :] = 0
    img[:, -margin:] = 0
    return img

def pipeline(img):
    global transform
    global roi
    roi_img = copy.deepcopy(img)
    roi_shown = draw_roi(roi_img, roi)
    cv2.namedWindow("roi")
    cv2.imshow("roi", roi_shown)
    filtered = filter(img)
    warped, undistorted_color = warp(filtered, img)
    cropped = crop_edges(warped, 50)
    result = find_lines(cropped, undistorted_color, np.linalg.inv(transform))
    cv2.waitKey(5)
    return result

def main():
    global cam
    global transform
    global warped_shape
    global roi
    cam = Camera()
    cam.read_chessboards('../camera_cal/calibration*.jpg')
    cam.calibrate_camera('../camera_cal/calibration*.jpg')
    roi = np.float32([[100, 666], [540, 500], [840, 500], [1230, 666]])
    warped_shape = np.array([800, 800])
    target_roi = np.float32([[0, warped_shape[0]], [0, 0], [warped_shape[0], 0], warped_shape])
    transform = cv2.getPerspectiveTransform(roi, target_roi)

    # # images = glob.glob('/home/mehdi/Udacity/CarND-Advanced-Lane-Lines/test_images/*.jpg')
    #images = ['/home/mehdi/Udacity/CarND-Advanced-Lane-Lines/test_images/test5.jpg']
    # for image in images:
    #     img = cv2.imread(image)
    #     filter_warp(img)

    videos = glob.glob('/home/mehdi/Udacity/CarND-Advanced-Lane-Lines/*.mp4')
    counter = 1
    for video in videos:
        video = "../challenge_video.mp4"
        clip1 = VideoFileClip(video)
        white_clip = clip1.fl_image(pipeline)
        white_clip.write_videofile("result_"+str(counter)+".mp4", audio=False)
        counter += 1
        #lt.show()
        #cv2.imshow("image", warped * 255)
        #cv2.waitKey(0)




if __name__=="__main__":
    main()




