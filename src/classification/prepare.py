"""
File: prepare.py
Description:
	Prepares the data for training of the object classification.
	Performs data augmentation of the original images of the objects and sets it up ready for training.

	By running the script with the '--inspect' flag the data that has been generated can instead be inspected to make
	sure that it looks good for training.
"""
import classification.data
import sys
import cv2
import numpy as np
import os
import pickle


obj_name2id = {
		"ball":        None,
		"bottle":      21,
		"box":         16,
		"brush":       22,
		"can":         15,
		"cutters":     17,
		"glass":       20,
		"hammer":      2,
		"knife":       12,
		"mug":         23,
		"pen":         3,
		"pitcher":     19,
		"scalp":       4,
		"scissors":    5,
		"screwdriver": 14,
		"tube":        18,
		}

with open("data/classification/clusters.pkl", "rb") as f:
	clusters = pickle.load(f)


def	cluster(oid):
	"""
	Return which cluster an object belongs to.
	"""
	for label in clusters:
		if oid in clusters[label]:
			return label
	return None


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
		prog = int(total / (len(filenames) * 3 * n * n) * 100)
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
		merged = classification.data.replace_with_depth(im, os.path.join(dir_depth, filename))
		out = classification.data.augment_image(merged, n=n, r=r)
		for i, image in enumerate(out):
			#cv2.imwrite(os.path.join(dst, "%s_%d.jpg" % (filename, i)), image.astype(np.uint8))
			filename = "{}_{}_{}_{}.npy".format(obj, oid, total, cluster(oid))
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
	for f in filenames:
		filepath = os.path.join(directory, f)
		#im = cv2.imread(filepath)
		im = np.load(filepath)
		cv2.imshow("opencv", im.astype(np.uint8))
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
		if quit:
			break


# settings
IMAGES_DIR = "data/classification/originals"
DATASET_DIR = "data/classification/images"
RADIUS = 40
N_AUGMENTATIONS = 5

if len(sys.argv) > 1 and sys.argv[1] == "--inspect":
	inspect_data(DATASET_DIR)
elif len(sys.argv) > 2 and sys.argv[1] == "--foo":
	d = "data/classification/originals/{}/depth".format(sys.argv[2])
	for f in os.listdir(d):
		print(os.path.join(d, f))
		im = classification.data.__load_depth__(os.path.join(d, f))
		im = im.reshape((424, 512))
		cv2.imshow("opencv", im.astype(np.uint8))
		while True:
			k = cv2.waitKey(0)
			if k == ord('s'):
				break
else:
	for name in os.listdir(IMAGES_DIR):
		# only traverse directories and
		# don't augment directories starting with 'new_', they are not part of the training set
		if name.startswith("new_") or not os.path.isdir(os.path.join(IMAGES_DIR, name)) or obj_name2id[name] is None:
			continue
		augment_directory(os.path.join(IMAGES_DIR, name), DATASET_DIR, n=N_AUGMENTATIONS, r=RADIUS)

cv2.destroyAllWindows()
