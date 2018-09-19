import prediction.data
import sys
import cv2
import numpy as np
import os

"""
def wait():
	while True:
		k = cv2.waitKey(0)
		if k == ord('q'):
			break


image_filename = "data/training/images/cnn/registered/0.jpg"
depth_filename = "data/training/images/cnn/depth/0"

image = cv2.imread(image_filename)
image = prediction.data.replace_with_depth(image, depth_filename)
images = prediction.data.augment_image(image, n=10)
for im in images:
	channels = [im[:, :, 0], im[:, :, 1], im[:, :, 2]]
	channels = list(map(lambda x: x.astype(np.uint8), channels))
	channels = cv2.hconcat(channels)
	cv2.imshow("opencv", channels)
	wait()

cv2.destroyAllWindows()
"""

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

	def print_progress():
		prog = int(total / (len(filenames) * 3 * n * n) * 100)
		prog_bar = ""
		prog_bar_len = 15
		for _ in range(int(prog/(100/prog_bar_len))):
			prog_bar += "#"
		for _ in range(prog_bar_len-int(prog/(100/prog_bar_len))):
			prog_bar += "-"

		print("\rAugmenting images found in '%s' to '%s': [%s] %d%%" % (src, dst, prog_bar, prog), end='', flush=True)

	for f in filenames:
		# make sure the file is a jpeg file
		filename, ext = os.path.splitext(f)
		if ext not in [".jpg", ".jpeg"]:
			continue

		# load registered image and depth image
		# swap the blue channel in the image with the depth and then augment this image
		# before storing to disk the newly created images
		im = cv2.imread(os.path.join(dir_registered, f))
		merged = prediction.data.replace_with_depth(im, os.path.join(dir_depth, filename))
		out = prediction.data.augment_image(merged, n=n, r=r)
		for i, image in enumerate(out):
			cv2.imwrite(os.path.join(dst, "%s_%d.jpg" % (filename, i)), image.astype(np.uint8))
			total += 1
			print_progress()

	print("\n==> %d images created" % total)


def inspect_data(directory):
	quit = False
	filenames = os.listdir(directory)
	for f in filenames:
		filepath = os.path.join(directory, f)
		im = cv2.imread(filepath)
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
		if quit:
			break


# settings
IMAGES_DIR = "data/training/images/cnn"
DATASET_DIR = "data/training/images/dataset"
RADIUS = 40
N_AUGMENTATIONS = 5

if len(sys.argv) > 1 and sys.argv[1] == "--inspect":
	inspect_data(DATASET_DIR)
else:
	augment_directory(IMAGES_DIR, DATASET_DIR, n=N_AUGMENTATIONS, r=RADIUS)

cv2.destroyAllWindows()
