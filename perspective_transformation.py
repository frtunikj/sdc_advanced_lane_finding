import pickle
import cv2
import glob
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from camera_calibration import undistortImage, calibrateCamera

def warpImage2birdsEyeView(img, img_size, perspective_M):

    # Use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, perspective_M, img_size, flags=cv2.INTER_LINEAR)
    
    return warped

def getPerspectiveTransform(img_size):
    
    src = np.float32([[567, 470],[717, 470],[1110, 720],[200, 720]])

    offset, mult = 100, 3    
    dst = np.float32([[mult * offset, offset],
                     [img_size[0] - mult * offset, offset],
                     [img_size[0] - mult * offset, img_size[1]],
                     [mult * offset, img_size[1]]])
    
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    return src, dst, M, M_inv

# Define a function that thresholds the L-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def hlsLThresh(img, thresh=(220, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hlsL = hls[:,:,1]
    hlsL = hlsL*(255/np.max(hlsL))
    # 2) Apply a threshold to the L channel
    binaryOutput = np.zeros_like(hlsL)
    binaryOutput[(hlsL > thresh[0]) & (hlsL <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binaryOutput

# Define a function that thresholds the B-channel of LAB
# Use exclusivexampleImg_LBThresh = lab_bthresh(exampleImg_unwarp, (min_b_thresh, max_b_thresh))
def labBThresh(img, thresh=(190,255)):
    # 1) Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    labB = lab[:,:,2]
    # don't normalize if there are no yellows in the image
    if np.max(labB) > 175:
        labB = labB*(255/np.max(labB))
    # 2) Apply a threshold to the L channel
    binaryOutput = np.zeros_like(labB)
    binaryOutput[((labB > thresh[0]) & (labB <= thresh[1]))] = 1
    # 3) Return a binary image of threshold result
    return binaryOutput

def pipelineImageTransformation(image):
    # Undistort
    ret, mtx, dist, rvecs, tvecs = pickle.load( open( "./camera_cal/camera_calibration_results.p", "rb" ) )
    undistImage = undistortImage(image, mtx, dist, plotImages=False)
    imageSize = (undistImage.shape[1], undistImage.shape[0]) 
    
    # Perspective Transform
    src, dst, perspective_M, perspective_M_inv = getPerspectiveTransform(imageSize)
    bidsEyeImage = warpImage2birdsEyeView(undistImage, imageSize, perspective_M)
    
    # HLS L-channel Threshold (using default parameters)
    imageLThresh = hlsLThresh(bidsEyeImage)

    # Lab B-channel Threshold (using default parameters)
    imageBThresh = labBThresh(bidsEyeImage)
    
    # Combine HLS and Lab B channel thresholds
    combined = np.zeros_like(imageBThresh)
    combined[(imageLThresh == 1) | (imageBThresh == 1)] = 1
    return combined 
