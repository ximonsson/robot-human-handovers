import cv2
import numpy as np
import sys

# constants for skin detection
# cr:  0.02 -> 0.18
# cb: -0.20 -> 0
ycrcb_min = np.array([80, 133, 77], np.uint8)
ycrcb_max = np.array([255, 173, 127], np.uint8)

# get image and convert to YCrCb color space
m = cv2.imread(sys.argv[1])
ycbcr = cv2.cvtColor(m, cv2.COLOR_BGR2YCR_CB)

# extract skin region by color
skin = cv2.inRange(ycbcr, ycrcb_min, ycrcb_max)

# erode and dilate back the extracted regions to filter out noise
kernel = np.ones((4, 4), np.uint8)
skin = cv2.erode(skin, kernel, iterations=1)
skin = cv2.dilate(skin, kernel, iterations=1)

# find contours around the remaining regions - this will be the grasping region
_, contours, hierarchy = cv2.findContours(skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
#m = cv2.drawContours(m, contours, contour, (0, 255, 0), 1)

# Draw the contour on the source image
#for i, c in enumerate(contours):
    #area = cv2.contourArea(c)
    #if area > 100:
        #m = cv2.drawContours(m, contours, i, (0, 255, 0), 1)

# draw rectangle around grasp region
orig = cv2.imread(sys.argv[2])
rx, ry, rw, rh = cv2.boundingRect(contour)
cv2.rectangle(orig, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 1)

#grasp_rect = cv2.minAreaRect(contour)
#cv2.drawContours(m, [np.int0(cv2.boxPoints(grasp_rect))], 0, (0, 255, 0), 1)

#grasp_ellipse = cv2.fitEllipse(contour)
#cv2.ellipse(m, grasp_ellipse, (255, 0, 0), 1)

# show the results
cv2.imshow("opencv grasp", orig)

while True:
    c = cv2.waitKey (0)
    if c == ord('q'):
        break
