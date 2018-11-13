"""
File: prepare.py
Description:
	Prepares the data for training of the object classification.
	Performs data augmentation of the original images of the objects and sets it up ready for training.

	By running the script with the '--inspect' flag the data that has been generated can instead be inspected to make
	sure that it looks good for training.
"""
from classification.data import augment_image, replace_with_depth
from classification.utils import find_arg
from classification import obj_name2id
import sys
import cv2
import numpy as np
import os
import pickle


def augment_directory(src, dst, n=10, r=20):
	"""
	Augment the images in src directory and output them to dst directory by k times.
	Data augmentation is done by random cropping and rotating the images, which are then
	stored in the image dimensions required by the AlexNet network.

	:param src: string - source directory containing images to augment.
	:param dst: string - destination directory to store the outputed images.
	:param n: integer - number of augmentations we should create per image.
	:param r: integer - radius around the center to perform augmentation.
	"""
	dir_registered = src + "/registered"
	dir_depth = src + "/depth"
	total = 0
	filenames = os.listdir(dir_registered)
	obj = os.path.split(src)[-1]
	oid = obj_name2id[obj]

	def print_progress():
		prog = int(total / (len(filenames) * 3 * (n * n + n + 1)) * 100)
		prog_bar = ""
		prog_bar_len = 15
		for _ in range(int(prog/(100/prog_bar_len))):
			prog_bar += "#"
		for _ in range(prog_bar_len-int(prog/(100/prog_bar_len))):
			prog_bar += "-"
		print("\rAugmenting images found in '%s': [%s] %d%%" % (src, prog_bar, prog), end='', flush=True)

	for f in filenames:
		# make sure the file is a jpeg file
		filename, ext = os.path.splitext(f)
		if ext not in [".jpg", ".jpeg"]:
			continue

		# load registered image and depth image
		# swap the blue channel in the image with the depth and then augment this image
		# before storing to disk the newly created images
		im = cv2.imread(os.path.join(dir_registered, f))
		merged = replace_with_depth(im, os.path.join(dir_depth, filename))
		out = augment_image(merged, n=n, r=r)
		for i, image in enumerate(out):
			filename = "{}_{}_{}.npy".format(obj, oid, total)
			outfile = os.path.join(dst, filename)
			np.save(outfile, image)
			total += 1
			print_progress()

	print(" ==> %d images created" % total)


def inspect_data(directory):
	"""
	Inspect the created dataset for the training of the object classification.
	"""
	quit = False
	filenames = os.listdir(directory)
	c = 1
	for f in filenames:
		filepath = os.path.join(directory, f)
		im = np.load(filepath)
		im = cv2.putText(
				im,
				"{}/{}".format(c, len(filenames)),
				(10, 10),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.25,
				(255, 255, 255))
		im = cv2.putText(
				im,
				"{}".format(f),
				(10, 20),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.25,
				(255, 255, 255))
		cv2.imshow("opencv", im)
		while True:
			k = cv2.waitKey(0)
			if k == ord('d'):
				os.remove(filepath)
				break
			elif k == ord('s'):
				break
			elif k == ord('q'):
				quit = True
				break
		c += 1
		if quit:
			break


# parse command line arguments for settings
IMAGES_DIR = find_arg("source", "data/classification/originals")
DATASET_DIR = find_arg("destination", "data/classification/images")
N_AUGMENTATIONS = int(find_arg("augmentations", "2"))
RADIUS = int(find_arg("radius", "5"))


#if any(map(lambda arg: arg=="--inspect", sys.argv)):
if "--inspect" in sys.argv:
	# we are only inspecting the dataset and maybe not keeping all of it
	inspect_data(DATASET_DIR)
else:
	for name in os.listdir(IMAGES_DIR):
		# only traverse directories and
		# don't augment directories starting with 'new_', they are not part of the training set
		#if name.startswith("new_") or \
		if not os.path.isdir(os.path.join(IMAGES_DIR, name)) or \
				obj_name2id[name] is None:
			continue
		augment_directory(os.path.join(IMAGES_DIR, name), DATASET_DIR, n=N_AUGMENTATIONS, r=RADIUS)

cv2.destroyAllWindows()
