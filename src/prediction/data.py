import os
import cv2
import struct
import numpy as np
from math import ceil
import alexnet
import random



def __crop__(im, c, r, s):
	# returns a random cropping of the image with center c and max radius away r
	if type(c) != np.ndarray:
		c = np.array(c)
	c += random.randint(-r, r)
	y1 = int(c[0]-s[0]/2)
	y2 = int(c[0]+s[0]/2)
	x1 = int(c[1]-s[1]/2)
	x2 = int(c[1]+s[1]/2)
	return im[y1:y2, x1:x2, :]


def __rotate__(im, c, s):
	# returns a random rotation of the image with center c
	a = random.randint(1, 180)
	R = cv2.getRotationMatrix2D(c, a, 1.0)
	return cv2.warpAffine(im, R, (im.shape[0], im.shape[1]))


def display(im):
	cv2.imshow("debug", im)
	while True:
		k = cv2.waitKey(0)
		if k == ord('q'):
			break


def augment_image(im, center=None, r=20, n=10, osize=(alexnet.IN_WIDTH, alexnet.IN_HEIGHT)):
	"""
	Augment an image a number of times using random cropping, rotation, flipping, etc.

	:param im: np.array - Image loaded with cv2.imread to perform augmentation on.
	:param center: tuple -
			Center around which to perform augmentation, if no value is passed
			the center of the image is chosen.
	:param r: integer - radius around center in which to perform the augmentation
	:param n: integer - Number of outputs
	:param osize: tuple of integers -
			Dimensions of the output images. Defaults to the size for the AlexNet network.
	:returns: list of images with length k with the outputed images
	"""
	if center is None:
		center = np.array(im.shape) / 2
		center = (center[0], center[1])

	images = []
	for _ in range(n):
		rotated = __rotate__(im, center, osize)
		for _ in range(n):
			images.append(__crop__(rotated, center, r, osize))

	return images


def __load_depth__(filepath):
	# load depth channel from file with filepath and returns as a np.array of floats
	with open(filepath, "rb") as f:
		data = f.read()
		count = ceil(len(data) / np.dtype(np.float32).itemsize)
		depth = np.zeros((count,), dtype=np.float32)
		iterable = struct.iter_unpack('f', data)
		for i, v in enumerate(iterable):
			depth[i] = v[0]
		return depth
	return None


def __merge_depth__(image, depth):
	# replace the blue channel in the image with the depth data and return the new image
	depth = depth.reshape((image.shape[0], image.shape[1]))
	im = np.copy(image).astype(np.float32)
	im[:, :, 0] = depth
	return im


def replace_with_depth(im, depth_filename):
	depth = __load_depth__(depth_filename)
	merged = __merge_depth__(im, depth)
	return merged


def augment_directory(src, dst, n=10):
	"""
	Augment the images in src directory and output them to dst directory by k times.
	Data augmentation is done by random cropping and rotating the images, which are then
	stored in the image dimensions required by the AlexNet network.

	:param src: string - source directory containing images to augment.
	:param dst: string - destination directory to store the outputed images.
	:param n: integer - number of augmentations we should create per image.
	:returns: integer - the total number of images created
	"""
	dir_registered = src + "/registered"
	dir_depth = src + "/depth"
	total = 0

	for f in os.listdir(dir_registered):
		# make sure the file is a jpeg file
		filename, ext = os.path.splitext(f)
		if ext not in [".jpg", ".jpeg"]:
			continue

		# load registered image and depth image
		# swap the blue channel in the image with the depth and then augment this image
		# before storing to disk the newly created images
		im = cv2.imread(os.path.join(dir_registered, f))
		depth = __load_depth__(os.path.join(dir_depth, filename))
		merged = __merge_depth__(im, depth)
		out = augment_image(merged, n=n)
		for i, image in enumerate(out):
			cv2.imwrite(os.path.join(dst, "%s_%d.jpg"))
			total += 1

	return total
