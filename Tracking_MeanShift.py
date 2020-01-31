import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import *


# Algorithm improvements
updating           = 0
frame_masking      = 1
hsv_input          = 0
gradient_computing = 1
GRADIENT_CHANNEL   = 2
GRADIENT_THRESHOLD = 50.

first_frame_show = 0
video_show       = 1

roi_defined = 0

def define_ROI(event, x, y, flags, param):
    global r, c, w, h, roi_defined
    # if the left mouse button was clicked, 
    # record the starting ROI coordinates 
    if event == cv2.EVENT_LBUTTONDOWN:
        r = x
        c = y
        roi_defined = 0
    # if the left mouse button was released,
    # record the ROI coordinates and dimensions
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
    # Interest measures
    # Can be useful to calibrate the min_hsv and max_hsv arrays
    cv2.imshow("First Frame S", first_hsv[:, :, 1])
    cv2.imshow("First Frame V", first_hsv[:, :, 2])

    first_frame_hist_S = cv2.calcHist([first_hsv], [1], None, [180], [0, 180])
    first_frame_hist_V = cv2.calcHist([first_hsv], [2], None, [180], [0, 180])
    plt.subplot(1, 2, 1)
    plt.title("First Frame S histogram")
    plt.plot(first_frame_hist_S)
    plt.subplot(1, 2, 2)
    plt.title("First Frame V histogram")
    plt.plot(first_frame_hist_V)
    plt.show()

    s_min = float(input("Type the minimum value of SATURATION:\n"))
    s_max = float(input("Type the maximum value of SATURATION:\n"))
    v_min = float(input("Type the minimum value of VALUE:\n"))
    v_max = float(input("Type the maximum value of VALUE:\n"))


# computation mask of the histogram:
# Pixels with S<30 or V<20 or V>235 are ignored 
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


# draw the initial backproject
while first_frame_show:
    first_frame_mask    = cv2.inRange(first_hsv, min_hsv, max_hsv)
    cv2.normalize(first_frame_mask, first_frame_mask, 0, 1, cv2.NORM_MINMAX)

    first_hsv[:, :, 0] *= first_frame_mask
    cv2.imshow('First frame H Masked', first_hsv[:, :, 0])
    
    first_dst = cv2.calcBackProject([first_hsv], [0], roi_hist, [0,180], 1)
    cv2.imshow('Initial Backproject', first_dst)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


# Setup the termination criteria: either 10 iterations,
# or move by less than 1 pixel
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1 )

#---------------------------------------------------------------------------------------------

if video_show:
    cpt = 1
    while(1):
        ret, current_frame  = cap.read()

        if ret == True:		
            current_frame_clone = current_frame.copy()

            frame_hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)

            if gradient_computing:
                Dx = cv2.Sobel(frame_hsv[:, :, GRADIENT_CHANNEL], cv2.CV_64F, 1, 0)
                Dy = cv2.Sobel(frame_hsv[:, :, GRADIENT_CHANNEL], cv2.CV_64F, 0, 1)
                G  = np.sqrt(Dx*Dx + Dy*Dy)

                cv2.normalize(G, G, 0, 255, cv2.NORM_MINMAX)
                G_mask = cv2.inRange(G, np.array([GRADIENT_THRESHOLD]), np.array([255.]))

                O     = np.arctan2(Dy, Dx)
                O_cos = 0.5*(np.cos(O) + 1)*G_mask #normalized to [0, 255]
                O_sin = 0.5*(np.sin(O) + 1)*G_mask #normalized to [0, 255]

                Orientation = np.zeros((O.shape[0], O.shape[1], 3))
                Orientation[:, :, 0] = O_cos
                Orientation[:, :, 1] = O_sin
                Orientation[:, :, 2] = 255 - G_mask

                cv2.imshow('Orientations', Orientation)

                hough = compute_hough(frame_hsv, O, G_mask, r_table)
                cv2.normalize(hough, hough, 0, 255, cv2.NORM_MINMAX)
                cv2.imshow('Hough Transform', hough)
                

            if frame_masking:
                frame_mask    = cv2.inRange(frame_hsv, min_hsv, max_hsv)
                cv2.normalize(frame_mask, frame_mask, 0, 1, cv2.NORM_MINMAX)
                frame_hsv[:, :, 0] *= frame_mask
                #cv2.imshow('Frame Mask', frame_mask)
                #cv2.imshow('Frame H', hsv[:, :, 0])

            
            # Backproject the model histogram roi_hist onto the
            # current image hsv, i.e. dst(x, y) = roi_hist(hsv(0, x, y))
            dst = cv2.calcBackProject([frame_hsv], [0], roi_hist, [0,180], 1)

            # Draw the backproject of the current image
            #cv2.imshow('Backproject', dst)

            # apply meanshift to dst to get the new location
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)

            # Draw a blue rectangle on the current image
            x, y, w, h    = track_window
            frame_tracked = cv2.rectangle(current_frame, (x, y), (x+w, y+h), (255, 0, 0) ,2)
            cv2.imshow('Sequence', frame_tracked)

            if updating:
                roi      = current_frame_clone[y:y+h, x:x+w]
                roi_hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                roi_mask = cv2.inRange(roi_hsv, min_hsv, max_hsv)
                roi_hist = cv2.calcHist([roi_hsv], [0], roi_mask, [180], [0,180])
                cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
                cv2.normalize(roi_mask, roi_mask, 0, 1, cv2.NORM_MINMAX)
                
                roi_hsv[:, :, 0] *= roi_mask
                cv2.imshow("ROI", roi)
                cv2.imshow('ROI HSV', roi_hsv[:, :, 0])


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
