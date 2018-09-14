import prediction.data
import sys
import cv2


def wait():
	while True:
		k = cv2.waitKey(0)
		if k == ord('q'):
			break


image_filename = "data/training/cnn/registered/0.jpg"
depth_filename = "data/training/cnn/depth/0"

im = cv2.imread(image_filename)
cv2.imshow("opencv", im)
wait()


im = prediction.data.replace_with_depth(im, depth_filename)
images = prediction.data.augment_image(im, n=10)

for im in images:
	cv2.imshow("opencv", im)
	wait()

cv2.destroyAllWindows()
