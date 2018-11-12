import classification.data as data
import sys
import os
import numpy as np
import cv2


rgbfile = sys.argv[1]
depthfile = sys.argv[2]

rgb = cv2.imread(rgbfile)

depth = data.__load_depth__(depthfile)
depth = depth.reshape((rgb.shape[0], rgb.shape[1]))
cv2.normalize(depth, depth, alpha=1.0, beta=0.0, norm_type=cv2.NORM_INF)
cv2.imshow("opencv", depth)
depth *= 255
cv2.imwrite("depth.jpg", depth.astype(np.uint8))
while True:
	k = cv2.waitKey(0)
	if k == ord('n'):
		break

rgbd = data.replace_with_depth(rgb, depthfile)
cv2.imshow("opencv", rgbd)
rgbd *= 255
cv2.imwrite("rgbd.jpg", rgbd.astype(np.uint8))
while True:
	k = cv2.waitKey(0)
	if k == ord('n'):
		break
