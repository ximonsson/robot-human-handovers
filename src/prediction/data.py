import os
import cv2
import struct
import numpy as np
from math import ceil


def load_data():
	return {}


def augment_image(im, center=None, k=10):
	"""
	Augment an image a number of times using random cropping and rotation.

	:param im: np.array - Image loaded with cv2.imread to perform augmentation on.
	:param center: tuple -
			Center around which to perform augmentation, if no value is passed
			the center of the image is chosen.
	:param k: integer - Number of outputs
	:returns: list of images with length k with the outputed images
	"""
	if center is None:
		center = im.shape / 2
		center = (center[0], center[1])
	return []


def __load_depth__(filename):
	with open(filename, "rb") as f:
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


def augment(src, dst, k=10):
	"""
	Augment the images in src directory and output them to dst directory by k times.
	Data augmentation is done by random cropping and rotating the images, which are then
	stored in the image dimensions required by the AlexNet network.

	:param src: string - source directory containing images to augment.
	:param dst: string - destination directory to store the outputed images.
	:param k: integer - number of augmentations we should create per image.
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
		out = augment_image(merged, k=k)
		for i, image in enumerate(out):
			cv2.imwrite(os.path.join(dst, "%s_%d.jpg"))
			total += 1

	return total

