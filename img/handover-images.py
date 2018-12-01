import cv2
import sys
import handoverdata as hd
import json
from handoverdata.object import load_objects_database
import numpy as np


with open("data/progress.json.bckp") as f:
	prog = json.load(f)

OBJECT_DB = "data/objects/objects.db"
objects = load_objects_database(OBJECT_DB)

with open("data/raw") as f:
	n = 0
	for val in prog["valid"]:
		handover = hd.read_at(f, val)
		obj = objects[handover.tag.id]

		print(handover.tag)

		# draw tag in frame
		detection = handover.tag.draw(handover.im)
		roi = np.copy(detection[
				handover.tag.center[1]-250:handover.tag.center[1]+250,
				handover.tag.center[0]-250:handover.tag.center[0]+250,
				])

		# draw ROI in frame
		detection = cv2.rectangle(
				detection,
				(int(detection.shape[1]/2-250), int(detection.shape[0]/2-250)),
				(int(detection.shape[1]/2+250), int(detection.shape[0]/2+250)),
				(0, 0, 255),
				2,
				)

		# mask out object
		mask = obj.mask
		mask = cv2.warpPerspective(mask, handover.H, (roi.shape[0], roi.shape[1]))
		roi = cv2.bitwise_and(roi, mask)

		# draw grasp on object
		_, h = cv2.invert(handover.H)
		roi = cv2.warpPerspective(roi, h, (roi.shape[0], roi.shape[1]))
		roi = handover.grasp.draw(roi, thickness=2)
		roi = cv2.warpPerspective(roi, handover.H, (roi.shape[0], roi.shape[1]))

		# display everything
		cv2.imshow("opencv frame", detection)
		cv2.imshow("opencv roi", roi)

		while True:
			k = cv2.waitKey(0)
			if k == ord('n'):
				break
			elif k == ord('s'):
				cv2.imwrite("{}_frame.jpg".format(n), detection)
				cv2.imwrite("{}_obj.jpg".format(n), roi)
				break

		n += 1
