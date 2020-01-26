import numpy as np
import cv2
import matplotlib.pyplot as plt


# Algorithm improvements
updating      = True
frame_masking = True

roi_defined = False
video       = True

def define_ROI(event, x, y, flags, param):
    global r, c, w, h, roi_defined
    # if the left mouse button was clicked, 
    # record the starting ROI coordinates 
    if event == cv2.EVENT_LBUTTONDOWN:
        #r, c = x, y
        r = x
        c = y
        roi_defined = False
    # if the left mouse button was released,
    # record the ROI coordinates and dimensions
    elif event == cv2.EVENT_LBUTTONUP:
        r2, c2 = x, y
        w = abs(r2 - r)
        h = abs(c2 - c)
        r = min(r, r2)
        c = min(c, c2)  
        roi_defined = True

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
#cv2.imshow('ROI', roi)

#---------------------------------------------------------------------------------------------

# conversion to Hue-Saturation-Value space
# 0 < H < 180 ; 0 < S < 255 ; 0 < V < 255
roi_hsv   = cv2.cvtColor(roi,               cv2.COLOR_BGR2HSV)
first_hsv = cv2.cvtColor(first_frame_clone, cv2.COLOR_BGR2HSV)

# Interest measures
# Can be useful to calibrate the min_hsv and max_hsv arrays
#cv2.imshow("H", first_hsv[:, :, 0])
#cv2.imshow("S", first_hsv[:, :, 1])
#cv2.imshow("V", first_hsv[:, :, 2])

# computation mask of the histogram:
# Pixels with S<30 or V<20 or V>235 are ignored 
min_hsv          = np.array([0., 40., 0.])
max_hsv          = np.array([180., 255., 235.])

roi_mask         = cv2.inRange(roi_hsv, min_hsv, max_hsv)
first_frame_mask = cv2.inRange(first_hsv, min_hsv, max_hsv)

#cv2.imshow('ROI Mask', roi_mask)
#cv2.imshow('First frame Mask', first_frame_mask)


# Marginal histogram of the Hue component
roi_hist = cv2.calcHist([roi_hsv], [0], roi_mask, [180], [0, 180])


# Histogram values are normalised to [0,255]
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)


plt.show()

# draw the initial backproject
while True:
    first_hsv[:, :, 0] *= first_frame_mask
    first_dst = cv2.calcBackProject([first_hsv], [0], roi_hist, [0,180], 1)
    cv2.imshow('Initial Backproject', first_dst)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


# Setup the termination criteria: either 10 iterations,
# or move by less than 1 pixel
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1 )

#---------------------------------------------------------------------------------------------

if video:
    cpt = 1
    while(1):
        ret, current_frame = cap.read()

        if ret == True:		
            hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)

            if frame_masking:
                frame_mask    = cv2.inRange(hsv, min_hsv, max_hsv)
                hsv[:, :, 0] *= frame_mask
                cv2.imshow("Frame HSV", hsv[:, :, 0])

            # Backproject the model histogram roi_hist onto the
            # current image hsv, i.e. dst(x, y) = roi_hist(hsv(0, x, y))
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)

            # Draw the backproject of the current image
            cv2.imshow('Backproject', dst)

            # apply meanshift to dst to get the new location
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)

            # Draw a blue rectangle on the current image
            x, y, w, h    = track_window
            frame_tracked = cv2.rectangle(current_frame, (x, y), (x+w, y+h), (255, 0, 0) ,2)
            cv2.imshow('Sequence', frame_tracked)

            if updating:
                roi      = current_frame[y:y+h, x:x+w]
                roi_hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                #roi_mask = cv2.inRange(roi_hsv, min_hsv, max_hsv)
                roi_hist = cv2.calcHist([roi_hsv], [0], None, [180], [0,180])
                cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
                #roi_hsv[:, :, 0] *= roi_mask
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
