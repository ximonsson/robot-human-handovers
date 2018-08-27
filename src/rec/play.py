import cv2
import sys
import os
import time

ROI_W = 350
ROI_H = 400
ROI_X = -int(ROI_W / 2)
ROI_Y = -100

directory = sys.argv[1]
files = os.listdir(directory)
files = sorted(files, key=lambda x: int(x.replace(".jpg", "")))
cv2.namedWindow("opencv playback")

for filename in files:
	im = cv2.flip(cv2.imread(os.path.join(directory, filename)), 1)
	cv2.rectangle(
		im,
		(int(im.shape[1]/2+ROI_X), int(im.shape[0]/2+ROI_Y)),
		(int(im.shape[1]/2+ROI_X+ROI_W), int(im.shape[0]/2+ROI_Y+ROI_H)),
		(0, 0, 255),
		2)
	cv2.imshow("opencv playback", im)
	k = cv2.waitKey(50)
	if k == ord('q'):
		break

cv2.destroyAllWindows()
