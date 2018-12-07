import os
import cv2
import numpy as np

filepath = 'results/classification/LR-0.0001__EP-20__BS-128__K-5__D-0.5_w2-t-0/bad_images.dat'
with open(filepath) as f:
	images = set([im.strip() for im in f.readlines()])

print("{} images were poorly predicited".format(len(images)))

for im in images:
	image = np.load(im)

	filename = os.path.basename(im)
	filename, _ = os.path.splitext(filename)
	print(im, filename)

	image = (image * 255.0).astype(np.uint8)
	'''
	cv2.imshow("opencv", image)
	while True:
		k = cv2.waitKey(0)
		if k == ord('n'):
			break
	'''
	cv2.imwrite("img/results/bad-images/{}.jpg".format(filename), image)
