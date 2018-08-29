from handoverdata.grasp import Grasp
from handoverdata.object import Object
from handoverdata.handover import Handover
import numpy as np
import cv2


DATA_NLINES = 4


def parse_data(data):
	"""
	Parse handover data connected to a frame.
	Returns the filepath to frame, AprilTag ID, Homography matrix and Grasp.
	"""
	lines = data.split("\n")

	# tag ID
	tid = lines[1].split(":")[0]

	# homograpy matrix
	h = lines[2][:-1].split(",")
	H = np.zeros((3, 3), np.float64)
	for i in range(len(h)):
		H[int(i/3)][i%3] = np.float64(h[i])

	# grasp region
	grasp_data = list(map(np.float64, lines[3].split(",")))
	g = Grasp(grasp_data[0], grasp_data[1], grasp_data[2], grasp_data[3], grasp_data[4])

	return Handover(lines[0][1:], int(tid), g, H)


def read_data(f):
	"""
	Reads data for one handover from the file pointer f.
	Note that this advances the pointer with DATA_NLINES
	"""
	data = ""
	for i in range(DATA_NLINES):
		data += f.readline()

	if len(data.split("\n")) < DATA_NLINES:
		return None
	return parse_data(data)


def rotation_matrix(H):
	"""
	Returns the rotation matrix from the homography matrix H
	"""
	# Normalization to ensure that ||c1|| = 1
	norm = np.sqrt(np.sum(H[:,0] ** 2))
	H /= norm
	c1 = H[:, 0]
	c2 = H[:, 1]
	c3 = np.cross(c1, c2)

	# create rotation matrix
	# calculate the rotation in Z-axis and rotate the original image
	R = np.zeros((3, 3), dtype=np.float64)
	for i in range(3):
		R[i, :] = [c1[i], c2[i], c3[i]]
	w, u, t = cv2.SVDecomp(R)
	return np.dot(u, t)
