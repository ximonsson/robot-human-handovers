"""
File: data.py
Description:
	Functions for manipulating and augmenting data in need to train for object classification
	by handover type.
"""
import cv2
import struct
import numpy as np
from math import ceil
import alexnet
import random
import os



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
	a = random.randint(1, 359)
	R = cv2.getRotationMatrix2D(c, a, 1.0)
	return cv2.warpAffine(im, R, (im.shape[0], im.shape[1]))


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
	def fn(im):
		for _ in range(n):
			rotated = __rotate__(im, center, osize)
			for _ in range(n):
				images.append(__crop__(rotated, center, r, osize))

	# augment on original image, flipped image on x-axis, and flipped image on y-axis
	fn(im)
	fn(cv2.flip(im, 0))
	fn(cv2.flip(im, 1))
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
	"""
	Replace the blue channel with the depth loaded from file on disk.
	:params im: np.ndarray - original image
	:params depth_filename: String - filepath on disk to file with binary depth image
	:returns: np.ndarray - new image with replaced blue channel (first one)
	"""
	depth = __load_depth__(depth_filename)
	merged = __merge_depth__(im, depth)
	return merged


def batches(data, size, imdim, outputs):
	"""
	Creates a generator to iterate over the batches of the given dataset yielding a batch
	of input data and the expected output.

	:param data: array - list of strings with filepaths to image files for the training
	:param size: integer - size of each batch
	:param imdim: array - dimensions of the images in the dataset
	:param outputs: integer - number of outputs of the network
	"""
	i = 0
	dim = [size]
	dim.extend(imdim)
	for _ in range(np.int(np.floor(len(data)/size))):
		x = np.ndarray(dim)
		y = np.zeros((size, outputs))
		for j in range(size):
			print(data[i])
			name, _ = os.path.splitext(os.path.basename(data[i]))
			cluster = np.int(name.split("_")[-1])
			x[j] = np.load(data[i])
			y[j][cluster] = 1
			i += 1
		yield x, y
