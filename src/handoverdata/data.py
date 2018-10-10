from .grasp import Grasp
from .handover import Handover
from .tag import TagDetection
import numpy as np


DATA_NLINES = 4


def parse_data(data):
	"""
	Parse handover data connected to a frame.

	:params data: lines read from file including handover data
	:returns: Handover object with the data
	"""
	lines = data.split("\n")

	# tag ID
	tag_data = lines[1].split(":")
	tid = tag_data[0]
	corners = []
	for c in tag_data[1][1:-1].split(")("):
		corner = eval("({})".format(c))
		corners.append(tuple(map(int, corner)))
	detection = TagDetection(int(tid), corners, tuple(map(int, eval(tag_data[2]))))

	# homograpy matrix
	h = lines[2][:-1].split(",")
	H = np.zeros((3, 3), np.float64)
	for i in range(len(h)):
		H[int(i/3)][i%3] = np.float64(h[i])

	# grasp region
	grasp_data = list(map(np.float64, lines[3].split(",")))
	g = Grasp(grasp_data[0], grasp_data[1], grasp_data[2], grasp_data[3], grasp_data[4])

	return Handover(lines[0][1:], detection, g, H)


def read_data(f):
	"""
	Reads data for one handover from the file pointer f.
	Note that this advances the pointer with DATA_NLINES

	:param f: file-pointer to datafile
	:returns: Handover object
	"""
	data = ""
	for i in range(DATA_NLINES):
		data += f.readline()

	if len(data.split("\n")) < DATA_NLINES:
		return None
	return parse_data(data)


def read_at(f, i):
	"""
	Read data a index.
	The file pointer is rewinded to the beginning of the data file and forwarded
	to the handover data at index. When the function returns the file pointer points
	to the data value after the supplied index.

	:param f: file pointer to data file
	:param i: handover index data
	:returns: handover data at index i
	"""
	f.seek(0)
	for _ in range(i * DATA_NLINES):
		f.readline()
	return read_data(f)
