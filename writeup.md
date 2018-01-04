# **Advanced Lane Finding**

[//]: # (Image References)

[image1]: ./output_images/undistort_chess.png 
[image2]: ./output_images/chess_with_points.png 
[image3]: ./output_images/undistorted_image.png 
[image4]: ./output_images/combined_S_and_X.png 
[image5]: ./output_images/combined_S_and_All.png
[image6]: ./output_images/X_gradient.png 
[image7]: ./output_images/Y_gradient.png 
[image8]: ./output_images/magnitude_gradient.png 
[image9]: ./output_images/direction_gradient.png
[image10]: ./output_images/S_threshold.png 
[image11]: ./output_images/points_perspective_transform.png 
[image12]: ./output_images/topview_straight_lines2.png
[image13]: ./output_images/topview_test2.png 
[image14]: ./output_images/topview_test3.png 
[image15]: ./output_images/topview_test4.png
[image16]: ./output_images/undistored_warp.png
[image17]: ./output_images/projected_lanes.png 
[image18]: ./output_images/window_detect.png 
[image19]: ./output_images/image_color_experiment.png 
[image20]: ./output_images/color_experiment.png 
[image21]: ./output_images/Lchannel.png 
[image22]: ./output_images/Bchannel.png 
[image23]: ./output_images/allTestFigures.png 

### Goals of the project
The goals / steps of this project were the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

### Repository Files Description

This repository includes the following files:

* [01_camera_calibration.ipynb](https://github.com/frtunikj/sdc_advanced_lane_finding/blob/master/01_camera_calibration.ipynb) - computes camera calibration matrix and distortion coefficients of the camera lens used given a set of chessboard images taken by the same (Udacity) camera.
* [02_color_transform_gradient_thershold.ipynb](https://github.com/frtunikj/sdc_advanced_lane_finding/blob/master/02_color_transform_gradient_thershold.ipynb) - uses color transforms, and sobel algorithm to create a thresholded binary image that has been filtered out of unnecessary information on the image.
* [03_perspective_transformation.ipynb](https://github.com/frtunikj/sdc_advanced_lane_finding/blob/master/03_perspective_transformation.ipynb) - applies perspective transform to see a “birds-eye view” of the image.
* [04_lane_detection.ipynb](https://github.com/frtunikj/sdc_advanced_lane_finding/blob/master/04_lane_detection.ipynb) -  detects lane pixels, determines curvature of the lanes, projects detected lane boundaries back onto the undistorted image of the original view.
* [05_lane_finding_video.ipynb](https://github.com/frtunikj/sdc_advanced_lane_finding/blob/master/05_lane_finding_video.ipynb) - outputs a video that contains the detected lanes/lane boundaries and other related information.

### Camera Calibration

The camera calibration code can be found under [01_camera_calibration.ipynb](https://github.com/frtunikj/sdc_advanced_lane_finding/blob/master/01_camera_calibration.ipynb) notebook (Step 2). 

In order to perform camera calibration one has to determine camera matrix, which captures the transformation between real-world 3D coordinates of objects and their corresponding 2D image coordinates. Commonly used approach is to use a checkerboard as an object, since it has a simple pattern with good contrast and known dimensions. The internal corners of the checkerboard are used to determine the 3D world and 2D image coordinates. Using cv2.findChessboardCorners(), the corners points are stored in an array imgpoints for each calibration image where the chessboard were found. 

![alt text][image2]

NOTE: For some of the test images, findChessboardCorners is not able to detect the desired number of internal corners (9x6) because it is not existant in the image.

The object points will always be the same as the known coordinates of the chessboard with zero as Z coordinate because the chessboard is flat. The object points are stored in an array called objpoints. The output objpoints and imgpoints are used to compute the camera calibration and distortion coefficients using the cv2.calibrateCamera() function. 

The distortion correction was applied to a test image using the cv2.undistort() function and the obtained results are follwing:

![alt text][image1]

### Pipeline (single images/video frames)

#### 1. Distortion-corrected image

Once the camera calibration is available from the previous step, it can be used to undistort real-world test images. The code for undistortion can be found under Step 3 of [01_camera_calibration.ipynb](https://github.com/frtunikj/sdc_advanced_lane_finding/blob/master/01_camera_calibration.ipynb) notebook. 

The image below depicts the results of applying undistort to one of the project images:

![alt text][image3]

The effect of undistort is subtle, but can be perceived from the difference in shape of the deer-warning road-sign i.e. appears flatter in the undistorted image.

#### 2. Binary image creation (color transforms, gradients)

The sobel operator is used to calculate the X or Y gradients (see Step 1 and 2 in see Step 6 in [02_color_transform_gradient_thershold.ipynb](https://github.com/frtunikj/sdc_advanced_lane_finding/blob/master/02_color_transform_gradient_thershold.ipynb)). Applying the sobel operator on an image is a way of
taking the derivative/gradient of the image in the X and the Y directions. This derivative measures how fast certain value changes from one pixel location to another.The magnitude and direction gradients were derived from the X and  Y gradients. Since the lane lines are more or less vertical in the camera images, the X-gradient captures them most clearly and that is why the other gradients were not used further. This can be seen also on the figures below that show the X, Y, magnitude and direction gradient separately. 

![alt text][image6]

![alt text][image7]

![alt text][image8]

![alt text][image9]

The sobel operator also takes a kernel parameter which changes the size of the region you apply the gradient to in an image. A low kernel size can have small noisy pixels but a high one can be prone to unwanted regions being part of the output. A kernel size of 3 was used for X, Y, magnitude gradient, and 15 for the direction gradient.

The goal of color thresholding is to mask everything out except yellows and whites. Experimnets in different color spaces were performed. The below image shows the various channels of the different color spaces for the same image (see Step 6 in [03_perspective_transformation.ipynb](https://github.com/frtunikj/sdc_advanced_lane_finding/blob/master/03_perspective_transformation.ipynb)):

![alt text][image19]

![alt text][image20]

At the end, the L channel of the HLS color space was used to isolate white lines and the B channel of the LAB color space was used to isolate yellow lines (see Step 7 and 8 in [03_perspective_transformation.ipynb](https://github.com/frtunikj/sdc_advanced_lane_finding/blob/master/03_perspective_transformation.ipynb)). No gradient thresholds was used in the pipeline since the white and the yellow lines were detected quite well with the color thresholding. In addition, in order each channel to be minimally tolerant to changes in lighting a normalization of the maximum values of the HLS L channel and the LAB B channel to 255 was performed. Figures below show examples of thresholds in the HLS L channel and the LAB B channel:

![alt text][image21]

![alt text][image22]

The final pipeline for processing images pipelineImageTransformation() i.e. applying color threshold can be found in Step 9 in [03_perspective_transformation.ipynb](https://github.com/frtunikj/sdc_advanced_lane_finding/blob/master/03_perspective_transformation.ipynb). The pipeline was also applied to all test images:

![alt text][image23]

#### 3. Perspective transform

An undistorted image of the vehicle’s perspective can be warped to output an image of another perspective such as a bird’s eye view from the sky by using a transformation matrix. A transformation matrix (perspective_M) can be calculated by giving the pixel coordinates of points of the input image of one perspective (src) and the corresponding pixels coordinates of the output perspective (dst) using the function getPerspectiveTransform() (see Step 4 in [03_perspective_transformation.ipynb](https://github.com/frtunikj/sdc_advanced_lane_finding/blob/master/03_perspective_transformation.ipynb)). The transformation matrix can be used to output an image to another perspective from a given perspective. 

To get the source points (src) and destination points (dst), an image in vehicle view of a road that has parallel lanes (not curved) was taken. Since the four destination points will form a rectangle as the output (lanes parallel), the source points were determined by experimenting (trial and error) and performing transformation (see Step 3 in [03_perspective_transformation.ipynb](https://github.com/frtunikj/sdc_advanced_lane_finding/blob/master/03_perspective_transformation.ipynb)). This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 567, 470      | 300, 100      | 
| 717, 470      | 980, 100      |
| 1110, 720     | 980, 720      |
| 200, 720      | 300, 720      |


To verify whether the perspective transform was working as expected, the src and dst points were drawn onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image16]

In addition, the perspective transform was applied/tested to streight and curved lane lines (see Step 5 in [03_perspective_transformation.ipynb](https://github.com/frtunikj/sdc_advanced_lane_finding/blob/master/03_perspective_transformation.ipynb)):

![alt text][image12]

![alt text][image13]

![alt text][image14]

![alt text][image15]


#### 4. Identifying lane line pixels and fitting their positions with a polynomial

To detect lane lines in the beginning i.e. in a first image (see detectLaneLines() in Step 2 of [04_lane_detection.ipynb](https://github.com/frtunikj/sdc_advanced_lane_finding/blob/master/04_lane_detection.ipynb)), first the X coordinates in the image that most likely coincide with the left and the right lane lines are calcutated. Ths is done by using the peaks of the histogram taken along the X-axis at the bottom of the image. Next, a “sliding window algorithm” (window on top of the other) is used to search windows around for X-coordinates and retrieve the pixel positions for the lane line pixels. The pixels inside a window are marked as pixels of interest and added to the list of points in the lane (see getLaneIndices() and getLanePixelPositions() in Step 1 of [04_lane_detection.ipynb](https://github.com/frtunikj/sdc_advanced_lane_finding/blob/master/04_lane_detection.ipynb)). These steps can be repeated until the top of the lane in an image is reached. In this way all collected pixels in the next step are fed to the polyfit() function which gives out the coefficients of the 2nd degree polynomial (x = y^2 + By + C). This is done in the function detectLaneLines() in Step 2 of [04_lane_detection.ipynb](https://github.com/frtunikj/sdc_advanced_lane_finding/blob/master/04_lane_detection.ipynb). 

![alt text][image18]

NOTE: The input to this stage is a binary image with a bird's eye view.

#### 5. Calculate radius of lane curvature and position of the vehicle with respect to center

The **radius of curvature** is calculated based upon this [website](https://www.intmath.com/applications-differentiation/8-radius-curvature.php) and the code can be found in Step 2 of [04_lane_detection.ipynb](https://github.com/frtunikj/sdc_advanced_lane_finding/blob/master/04_lane_detection.ipynb) i.e. function calculateAndWriteCurvatureRadius(). The X and Y coordinates of the lane lines are translated to their metric values using appropriate scaling factors provided in the project guidelines. Then a second-order polynomial fit is calculated on these metric values. The radius of curvature is then determined by using the formula:

```
curvatureRadius = (((1 + (2 * fit[0] * yEval * yMeterPerPixel + fit[1])**2)**1.5) / np.absolute(2*fit[0]))
```

Here, fit[0] is the first coefficient (the y-squared coefficient) of the second order polynomial fit, and fit[1] is the second (y) coefficient. yEval is the y position within the image upon which the curvature calculation is based. yMeterPerPixel is the factor used for converting from pixels to meters. Finally, the radius for the left and right lines is averaged in order to determine the radius of curvature for the lane.

The **position of the vehicle** with respect to the center of the lane is calculated with the following lines of code (function caculateAndWriteLaneOffset() in Step 2 of [04_lane_detection.ipynb](https://github.com/frtunikj/sdc_advanced_lane_finding/blob/master/04_lane_detection.ipynb)):

```
lane_mid_x = x_left + (x_right - x_left)/2
offset = x_meter_per_pixel * (img_mid_x - lane_mid_x)
```

As one can notice, this is done by taking the difference between the X coordinates of the midpoint of the determined lane lines and the center of the image. This assumes that the camera is mounted exactly along the center axis of the car.

#### 6. Example image with identified lane area 

The code that detects lane line and projects them on the original image can be found in Step 5 and 6 of [04_lane_detection.ipynb](https://github.com/frtunikj/sdc_advanced_lane_finding/blob/master/04_lane_detection.ipynb). A polygon is generated based on plots of the left and right fits, warped back to the perspective of the original image using the inverse perspective matrix Minv and overlaid onto the original image. The image below is an example of the results of the projectLaneLinesRoad() function:

![alt text][image17]

The image above, also shows the results of the calculateAndWriteCurvatureRadius() and caculateAndWriteLaneOffset() functions (Step 4 in [04_lane_detection.ipynb](https://github.com/frtunikj/sdc_advanced_lane_finding/blob/master/04_lane_detection.ipynb)), which calculates and writes text regarding the curvature of the lane and vehicle position with respect to center.

---

### Pipeline (video)

Once the lane lines are detected at the beginning i.e. in the first image/video frame (see above Step 4), the lane lines are tracked the subsequent images/frames by specifying a search window around the polynomial fit determined previously (see trackLaneLines() in Step 2 of [05_lane_finding_video.ipynb](https://github.com/frtunikj/sdc_advanced_lane_finding/blob/master/05_lane_finding_video.ipynb)). This saves an exhaustive search from bottom to top of the image as required during the line detection described above. For tracking lines across frames, an average over some previous fits is used to place the search window. Also, for plotting the lane line on the output image an average fit is used which includes the current fit (see function argument "useAverageFit"). 

The resulting video can be found under [output_project_video.mp4](https://github.com/frtunikj/sdc_advanced_lane_finding/blob/master/videos/output_project_video.mp4). The complete pipeline for generating the vide via processing video images/frames can be found [05_lane_finding_video.ipynb](https://github.com/frtunikj/sdc_advanced_lane_finding/blob/master/05_lane_finding_video.ipynb). 

---

### Discussion and potential for further improvements (undone work)

It was not easy to get the right parameters for gradient thresholding which is very important to get a curve that makes sense. However, after a lot of experimenting optimal parameters were found (at leat for the testing video). The image to image/frame to frame tracking fails for sharp curves in the challenge videos provided by Udacity. This is because the search window predicted from previous frames excludes sharply curving lane-lines deeper in the image. To overcome this problem, one might consider to include information from frame to frame for example, use the radius of curvature to correct the predicted search window. This could help to bend the search window and in that way to capture curved lane lines deeper into the image. In adition, one could consider to work on the issue when the lane line detection/tracking fails to work e.g. due to bad lane marking, weather conditions etc..