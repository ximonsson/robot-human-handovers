from handoverdata.grasp import Grasp
from handoverdata.handover import Handover
import numpy as np


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
