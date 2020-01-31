import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import *


# Algorithm improvements
hsv_input        = 0
updating         = 0
frame_masking    = 1
hough_tracker    = 1
argmax_computing = 1

GRADIENT_CHANNEL   = 2
GRADIENT_THRESHOLD = 50.
NUM_MAX_POINTS     = 2


roi_defined = 0

def define_ROI(event, x, y, flags, param):
    global r, c, w, h, roi_defined
    if event == cv2.EVENT_LBUTTONDOWN:
        r = x
        c = y
        roi_defined = 0
    elif event == cv2.EVENT_LBUTTONUP:
        r2, c2 = x, y
        w = abs(r2 - r)
        h = abs(c2 - c)
        r = min(r, r2)
        c = min(c, c2)  
        roi_defined = 1

#---------------------------------------------------------------------------------------------

cap = cv2.VideoCapture('./Sequences/Antoine_Mug.mp4')

# take first frame of the video
ret, first_frame = cap.read()

# load the image, clone it, and setup the mouse callback function
first_frame_clone = first_frame.copy()
cv2.namedWindow("First image")
cv2.setMouseCallback("First image", define_ROI)
 
# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("First image", first_frame)
    key = cv2.waitKey(1) & 0xFF

    # if the ROI is defined, draw it!
    if (roi_defined):
        # draw a green rectangle around the region of interest
        cv2.rectangle(first_frame, (r, c), (r+w, c+h), (0, 255, 0), 2)
    
    # else reset the image...
    else:
        first_frame = first_frame_clone.copy()
    
    # if the 'q' key is pressed, break from the loop
    if key == ord("q"):
        break

# set up the ROI for tracking 
track_window = (r, c, w, h)
roi          = first_frame_clone[c:c+h, r:r+w]

#---------------------------------------------------------------------------------------------

# conversion to Hue-Saturation-Value space
# 0 < H < 180 ; 0 < S < 255 ; 0 < V < 255
roi_hsv   = cv2.cvtColor(roi,               cv2.COLOR_BGR2HSV)
first_hsv = cv2.cvtColor(first_frame_clone, cv2.COLOR_BGR2HSV)

s_min = 50.
s_max = 255.
v_min = 5.
v_max = 100.

if hsv_input:
    # show the hsv components of the first frame to calibrate the bounds of the mask function
    s_min, s_max, v_min, v_max = hsv_bound_input(first_hsv)


# computation mask of the histogram:
min_hsv  = np.array([0., s_min, v_min])
max_hsv  = np.array([180., s_max, v_max])

roi_mask = cv2.inRange(roi_hsv, min_hsv, max_hsv)


# Marginal histogram of the Hue component
roi_hist = cv2.calcHist([roi_hsv], [0], roi_mask, [180], [0, 180])


# Histogram values are normalised to [0,255]
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)


# create RTable
cv2.normalize(roi_mask, roi_mask, 0, 1, cv2.NORM_MINMAX)
roi_hsv[:, :, GRADIENT_CHANNEL] *= roi_mask
r_table = construct_RTable(roi_hsv, GRADIENT_THRESHOLD)


# Setup the termination criteria: either 10 iterations,
# or move by less than 1 pixel
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1 )

#---------------------------------------------------------------------------------------------

cpt = 1
while(1):
    ret, current_frame  = cap.read()

    if ret == True:		
        current_frame_clone = current_frame.copy()

        frame_hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)

        Dx, Dy      = image_gradient(frame_hsv, channel=GRADIENT_CHANNEL)
        grad_module = gradient_module(Dx, Dy)

        #cv2.normalize(G, G, 0, 255, cv2.NORM_MINMAX)
        grad_mask = cv2.inRange(grad_module, np.array([GRADIENT_THRESHOLD]), np.array([np.inf]))

        orient = np.arctan2(Dy, Dx)
        orient_cos = 0.5*(np.cos(orient) + 1)*grad_mask #normalized to [0, 255]
        orient_sin = 0.5*(np.sin(orient) + 1)*grad_mask #normalized to [0, 255]

        orient_img = np.zeros((orient.shape[0], orient.shape[1], 3))
        orient_img[:, :, 0] = orient_cos
        orient_img[:, :, 1] = orient_sin
        orient_img[:, :, 2] = 255 - grad_mask

        cv2.imshow('Orientations', orient_img)

        hough = compute_hough(frame_hsv, orient, grad_mask, r_table).astype(float)
        cv2.normalize(hough, hough, 0, 255, cv2.NORM_MINMAX)
        cv2.imshow('Hough Transform', hough)

        if hough_tracker:
            if argmax_computing:
                track_window = hough_window(hough, num_points=NUM_MAX_POINTS)
            else:
                ret, track_window = cv2.meanShift(hough, track_window, term_crit)
        

        if frame_masking:
            frame_mask = cv2.inRange(frame_hsv, min_hsv, max_hsv)
            cv2.normalize(frame_mask, frame_mask, 0, 1, cv2.NORM_MINMAX)
            frame_hsv[:, :, 0] *= frame_mask
        
        # Backproject the model histogram roi_hist onto the
        # current image hsv, i.e. dst(x, y) = roi_hist(hsv(0, x, y))
        dst = cv2.calcBackProject([frame_hsv], [0], roi_hist, [0,180], 1)

        # Draw the backproject of the current image
        cv2.imshow('Backproject', dst)


        if not hough_tracker:
            # apply meanshift to dst to get the new location
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Draw a blue rectangle on the current image
        x, y, w, h    = track_window
        frame_tracked = cv2.rectangle(current_frame, (x, y), (x+w, y+h), (0, 0, 255) ,2)
        cv2.imshow('Sequence', frame_tracked)

        if updating:
            # Does not help...
            roi      = current_frame_clone[y:y+h, x:x+w]
            roi_hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            roi_mask = cv2.inRange(roi_hsv, min_hsv, max_hsv)
            roi_hist = cv2.calcHist([roi_hsv], [0], roi_mask, [180], [0,180])
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
            cv2.normalize(roi_mask, roi_mask, 0, 1, cv2.NORM_MINMAX)
            
            roi_hsv[:, :, 0] *= roi_mask


        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('Frame_%04d.png'%cpt, frame_tracked)
        cpt += 1
    else:
        break

cv2.destroyAllWindows()
cap.release()
