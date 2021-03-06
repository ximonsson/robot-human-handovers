"""
File: data.py
Description:
	Functions for manipulating and augmenting data in need to train for object classification
	by handover type. Also functions for load batches used for feeding a network during training
	and testing as well as split data into balanced datasets.
"""
import cv2
import struct
import numpy as np
from math import ceil
import classification.alexnet as alexnet
import random
import os
from classification import Object, __class_assignments__


def __crop__(im, center, radius, size):
	# returns a random cropping of the image with center c and max radius away r
	c = np.copy(center)
	c[0] += random.randint(-radius, radius)
	c[1] += random.randint(-radius, radius)
	y1 = int(c[0]-size[0]/2)
	y2 = int(c[0]+size[0]/2)
	x1 = int(c[1]-size[1]/2)
	x2 = int(c[1]+size[1]/2)
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
	:returns: list of images of length 3 * (n^2 + n + 1) with the outputed images
	"""
	if center is None:
		center = np.array(im.shape) / 2
		center = (center[0], center[1])

	images = []
	def __augment__(im):
		images.append(__crop__(im, center, 0, osize))
		for _ in range(n):
			rotated = __rotate__(im, center, osize)
			images.append(__crop__(rotated, center, 0, osize))
			for _ in range(n):
				images.append(__crop__(rotated, center, r, osize))

	# augment on original image, flipped image on x-axis, and flipped image on y-axis
	__augment__(im)
	__augment__(cv2.flip(im, 0))
	__augment__(cv2.flip(im, 1))
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
	cv2.normalize(im, im, alpha=1.0, beta=0.0, norm_type=cv2.NORM_INF)
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
	cv2.normalize(depth, depth, alpha=1.0, beta=0.0, norm_type=cv2.NORM_INF)
	merged = __merge_depth__(im, depth)
	return merged


def datasets(src, objects, k=1):
	"""
	Create training and validation datasets.
	Go through data files that are found in specified source directory and split by objects
	according to ratio.

	:param src: string - filepath to source directory with data
	:param objects: array - list of objects that we want included in the dataset
	:param k: integer - number of datasets to create.
	:returns: array of arrays - k arrays with balanced datasets.
	"""

	# list files in the source directory
	# sort out the files that belong to the supplied list of objects
	# start by balancing the images so there are an equal amount of images per object

	object_files = {o.ID: o.files(src) for o in objects}
	for v in object_files.values():
		random.shuffle(v)

	# divide into datasets per class and balance between them before returning
	# them divided into k different ones

	class_files = {
			c: sum([object_files[oID] for oID in o if oID in object_files], []) \
					for c, o in __class_assignments__.items()}

	# find the smallest class and rectify the number of images per object in the larger class to
	# to make it more balanced

	n = min(map(len, class_files.values()))
	for c in class_files:
		obj = [o for o in __class_assignments__[c] if o in object_files.keys()]
		nim = int(n / len(obj)) # number of images per object in the class for it to be balanced
		for o in obj:
			if o in object_files:
				object_files[o] = object_files[o][:nim]
		class_files[c] = sum([object_files[oID] for oID in obj if oID in object_files], [])
		random.shuffle(class_files[c])

	# create k datasets

	datasets = []
	n = min(map(len, class_files.values()))
	size = int(n / k)
	n = size * k
	for i in range(0, n, size):
		files = {c: f[i:i+size] for c, f in class_files.items()}
		datasets.append([os.path.join(src, f) for f in sum(files.values(), [])])

	return datasets


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
	size_ = len(data) if size > len(data) else size
	dim = [size_]
	dim.extend(imdim)
	for b in range(int(len(data)/size_)):
		x = np.ndarray(dim)
		y = np.zeros((size_, outputs))
		for j in range(size_):
			name, _ = os.path.splitext(os.path.basename(data[i]))
			name = name.split("_")[0]
			cl = Object(name).cl
			x[j] = np.load(data[i])

			# change base to [0, 255], RGB -> BGR, and scale to imagenet mean
			x[j] *= 255.0
			x[:, :, 0], x[:, :, 2] = x[:, :, 2], x[:, :, 0]
			x[j] -= np.array([104., 117, 124.], dtype=np.float32)

			y[j][cl] = 1
			i += 1
		yield b, x, y
