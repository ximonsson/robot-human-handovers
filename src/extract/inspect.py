import numpy as np
import cv2
import os
import sys
import math
import json


ROI_W = 350
ROI_H = 400
ROI_X = -ROI_W / 2
ROI_Y = -100


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

	def __dict__(self):
		return {
			"x": self.x,
			"y": self.y,
			"w": self.w,
			"h": self.h,
			"a": self.a,
			}


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
	item = cv2.flip(item, 1) # flip it because it is an image from the kinect
	# draw the grasping region
	box = cv2.boxPoints(g.box())
	box = np.int0(box)
	cv2.drawContours(item, [box], 0, (0, 0, 255))
	# warp it to the same perspective as in the handover
	#item = cv2.warpPerspective(item, H, (item.shape[0], item.shape[1]))

	# Normalization to ensure that ||c1|| = 1
	norm = np.sqrt(np.sum(H[:,0] ** 2))
	H /= norm
	c1 = H[:, 0]
	c2 = H[:, 1]
	c3 = np.cross(c1, c2)

	# create rotation matrix
	R = np.zeros((3, 3), dtype=np.float64)
	for i in range(3):
		R[i, :] = [c1[i], c2[i], c3[i]]
	w, u, t = cv2.SVDecomp(R)
	R = np.dot(u, t)

	# calculate rotation in Z-axis and rotate the original image
	# TODO fix to dynamic center relative to the center of the tag
	thetaZ = math.atan2(R[1,0], R[0,0])
	thetaZ = - thetaZ * 180 / math.pi
	rot = cv2.getRotationMatrix2D((item.shape[0]/2, item.shape[1]/2), thetaZ, 1.0)
	rotated = cv2.warpAffine(item, rot, (item.shape[0], item.shape[1]))
	# display everything
	cv2.imshow("opencv frame", cv2.flip(cv2.imread(f), 1))
	cv2.imshow("opencv data", rotated)



data_valid = {"data": []} # data that is assumed to be valid
data_discard = {"data": []} # data to be discarded
data_backlog = {"data": []} # data that is put in to a backlog to be checked over later again with different parameters maybe


def append_data(data, filepath, tag_id, H, grasp):
	data["data"].append(
			{
				"file": filepath,
				"grasp": grasp.__dict__(),
				"tag": {
					"id": tag_id,
					},
				"H": H.tolist(),
				}
			)


def store_data():
	# filepaths to where to store the different data
	DATA_FP_VALID = "data_valid.json"
	DATA_FP_DISCARD = "data_discard.json"
	DATA_FP_BACKLOG = "data_backlog.json"

	with open(DATA_FP_VALID, "w") as f:
		json.dump(data_valid, f)

	with open(DATA_FP_DISCARD, "w") as f:
		json.dump(data_discard, f)

	with open(DATA_FP_BACKLOG, "w") as f:
		json.dump(data_backlog, f)


DATA_FILE = sys.argv[1]
DATA_NLINES = 4

# if we supplied at what entry we want to start
start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
n = start
start *= 4

quit = False
# iterate over all files in directory of extracted data
with open(DATA_FILE, "r") as f:
	data = ""
	i = 0
	for line in f:
		if start != 0:
			start -= 1
			continue

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
				append_data(data_valid, fp, tid, H, g)
				break
			elif k == ord('d'):
				append_data(data_discard, fp, tid, H, g)
				break
			elif k == ord('a'):
				append_data(data_backlog, fp, tid, H, g)
				break

		if quit:
			break

		n += 1
		data = ""
		i = 0

print("finished at entry [%d]" % n)
cv2.destroyAllWindows()
store_data()
