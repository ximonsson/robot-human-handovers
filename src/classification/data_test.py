import cv2
import numpy as np
from classification.data import __merge_depth__, __load_depth__


def test_merging():
	rgb = np.array([
		[
			[1, 2, 3],
			[1, 2, 3],
			[1, 2, 3]],
		[
			[1, 2, 3],
			[1, 2, 3],
			[1, 2, 3]],
		[
			[1, 2, 3],
			[1, 2, 3],
			[1, 2, 3]]])
	depth = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
	out = __merge_depth__(rgb, depth)
	assert np.array_equiv(out[:, :, 0], depth.reshape(3, 3))


def test_loading_merging(color, depth):
	color = cv2.imread(color)
	d = __load_depth__(depth)
	out = __merge_depth__(color, d)
	with open(depth, "rb") as f:
		data = f.read()

	assert d.tobytes() == data
	assert out[:, :, 0].tobytes() == data


def test_loading_depth(filename):
	d = __load_depth__(filename)
	with open(filename, "rb") as f:
		data = f.read()
		assert d.tobytes() == data
	return d



COLOR_IMAGE = "data/objects/registered/0.jpg"
DEPTH_IMAGE = "data/objects/depth/0"

test_merging()
test_loading_depth(DEPTH_IMAGE)
test_loading_merging(COLOR_IMAGE, DEPTH_IMAGE)
print("All tests passed")
