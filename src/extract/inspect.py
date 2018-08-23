import numpy as np
import cv2
import os
import sys


ROI_W = 300
ROI_H = 300

class Grasp:
	"""
	Grasp region on an object.
	Represents a rotated angle.
	"""

	def __init__(self, x, y, w, h, a):
		self.x = x
		self.y = y
		self.w = w
		self.h = h
		self.a = a

	def __str__(self):
		return "(%f, %f): [%f x %f]: %f" % (self.x, self.y, self.w, self.h, self.a)

	def box(self):
		return (self.x, self.y), (self.w, self.h), self.a


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

	return lines[0][1:], int(tid), H, g


def display(f, tid, H, g):
	"""
	Display the handover data in form of the object, it's orientation during the handover
	and the grasping region.
	"""
	# load item image
	item = cv2.imread("data/objects/%d.jpg" % tid)
	item = cv2.resize(item, (ROI_W, ROI_H))
	item = cv2.flip(item, 1) # flip it because it is an image from the kinect
	# draw the grasping region
	box = cv2.boxPoints(g.box())
	box = np.int0(box)
	cv2.drawContours(item, [box], 0, (0, 0, 255))
	# warp it to the same perspective as in the handover
	item = cv2.warpPerspective(item, H, (item.shape[0], item.shape[1]))

	# display everything
	cv2.imshow("opencv frame", cv2.flip(cv2.imread(f), 1))
	cv2.imshow("opencv data", item)



data_valid = [] # data that is assumed to be valid
data_discard = [] # data to be discarded
data_backlog = [] # data that is put in to a backlog to be checked over later again with different parameters maybe

def store_data():
	# filepaths to where to store the different data
	DATA_FP_VALID = "data_valid"
	DATA_FP_DISCARD = "data_discard"
	DATA_FP_BACKLOG = "data_backlog"

	with open(DATA_FP_VALID, "w") as f:
		f.write("".join(data_valid))

	with open(DATA_FP_DISCARD, "w") as f:
		f.write("".join(data_discard))

	with open(DATA_FP_BACKLOG, "w") as f:
		f.write("".join(data_backlog))



DATA_FILE = sys.argv[1]
DATA_NLINES = 4
quit = False
# iterate over all files in directory of extracted data
with open(DATA_FILE, "r") as f:
	data = ""
	i = 0
	for line in f:
		data += line
		i += 1
		if not i == DATA_NLINES: # continue reading until we have data for an entire handover
			continue

		# parse data and display it
		# afterwards wait for keypress with command about what to do with the data
		fp, tid, H, g = parse_data(data)
		display(fp, tid, H, g)

		while True:
			k = cv2.waitKey(0)
			if k == ord('q'):
				quit = True
				break
			elif k == ord('s'):
				data_valid.append(data)
				break
			elif k == ord('d'):
				data_discard.append(data)
				break
			elif k == ord('a'):
				data_backlog.append(data)
				break

		if quit:
			break
		# reset
		data = ""
		i = 0

cv2.destroyAllWindows()
store_data()
