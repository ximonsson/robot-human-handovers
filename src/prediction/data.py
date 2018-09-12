import os
import cv2
import struct


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
	pass


def __merge_depth__(image, depth):
	# read depth data from the file
	# replace the blue channel in the image with the depth data and return the new image
	with open(depth, "rb") as f:
		data = f.read()
		depth = np.array([], dtype=np.float32)
		for v in struct.iter_unpack('f', data):
			depth = np.append(depth, v[0])

		depth.reshape(image.shape)
		im = np.copy(image)
		im[0, :, :] = depth

		return im

	return None


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
		merged = __merge_depth__(im, os.path.join(dir_depth, filename))
		out = augment_image(merged, k=k)
		for i, image in enumerate(out):
			cv2.imwrite(os.path.join(dst, "%s_%d")
			total += 1

	return total

