import cv2
import numpy as np


COLOR_IMAGE = "data/objects/registered/0.jpg"
DEPTH_IMAGE = "data/objects/depth/0"


def test_merging(color, depth):
	from data import __merge_depth__ as fn
	color = cv2.imread(color)
	with open(depth, "rb") as f:
		data = f.read()
	out = fn(color, depth)
	assert out[:, :, 0].tobytes() == data


test_merging(COLOR_IMAGE, DEPTH_IMAGE)
