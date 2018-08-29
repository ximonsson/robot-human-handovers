import numpy as np
import cv2
import os
import sys
import math
import json
import handoverdata as hd
from handoverdata.object import load_objects_database


ROI_W = 350
ROI_H = 400
ROI_X = -ROI_W / 2
ROI_Y = -100


objects = load_objects_database("data/objects/objects.db")


def display(handover):
	"""
	Display the handover data in form of the object, it's orientation during the handover
	and the grasping region.
	"""
	# load item image
	item = objects[handover.objectID]
	im = np.copy(item.im)

	# draw the grasping region
	box = cv2.boxPoints(handover.grasp.box())
	box = np.int0(box)
	cv2.drawContours(im, [box], 0, (0, 0, 255))
	# warp it to the same perspective as in the handover
	#item = cv2.warpPerspective(item, handover.H, (item.shape[0], item.shape[1]))

	H = np.copy(handover.H)
	R = hd.rotation_matrix(H)
	thetaZ = math.atan2(R[1,0], R[0,0]) # rotation in Z-axis
	thetaZ = - thetaZ * 180 / math.pi
	R = cv2.getRotationMatrix2D(item.center, thetaZ, 1.0)
	rotated = cv2.warpAffine(im, R, (im.shape[0], im.shape[1]))

	# display everything
	cv2.imshow("opencv frame", handover.im)
	cv2.imshow("opencv data", rotated)


"""
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
"""


DATA_RAW_FILE = sys.argv[1]
DATA_PROGRESS_FILE = "data/training/progress.json"

data_valid = []
data_discard = []
data_backlog = []
n = 0
start = 0

if os.path.isfile(DATA_PROGRESS_FILE):
	with open(DATA_PROGRESS_FILE) as f:
		data = json.load(f)
		data_valid = data["valid"]
		data_discard = data["discard"]
		data_backlog = data["backlog"]
		start = data["stopped"]

n = start
start *= hd.DATA_NLINES


def store_data():
	with open(DATA_PROGRESS_FILE, 'w') as f:
		data = {
				"stopped": n,
				"valid": data_valid,
				"discard": data_discard,
				"backlog": data_backlog, }
		json.dump(data, f)


quit = False
# iterate over all files in directory of extracted data
with open(DATA_RAW_FILE, "r") as f:
	# jump over lines until we get to the start index
	for i in range(start):
		f.readline()

	while not quit:
		# read data for a handover and display
		# wait for keypress about what to do with the data
		handover = hd.read_data(f)
		if handover is None:
			break

		display(handover)
		while True:
			k = cv2.waitKey(0)
			if k == ord('q'):
				quit = True
				break
			elif k == ord('s'):
				data_valid.append(n)
				break
			elif k == ord('d'):
				data_discard.append(n)
				break
			elif k == ord('a'):
				data_backlog.append(n)
				break

		if not quit:
			n += 1

print("finished at entry [%d]" % n)
cv2.destroyAllWindows()
store_data()
