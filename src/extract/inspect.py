import cv2
import os
import sys
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
	im = handover.draw(item)
	# display everything
	cv2.imshow("opencv frame", handover.im)
	cv2.imshow("opencv data", im)


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
