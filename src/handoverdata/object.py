import numpy as np
import cv2


class Object:
	"""
	Reference object.
	Contains information about filepath to image on disk, center point of the tag, corner
	points and tag ID.
	"""
	def __init__(self, filename, tid, center, corners, im=None):
		self.filename = filename
		self.tid = tid
		self.center = center
		self.corners = corners
		self.im = im

	def __str__(self):
		return "[%d] %s : %s : %s" % (self.tid, self.filename, self.center, self.corners)


def load_objects_database(filepath):
	"""
	Returns dictionary with objects loaded from database file at filepath.
	"""
	objects = dict()
	with open("data/objects/objects.db") as f:
		for line in f:
			tokens = line[:-1].split(":")
			# parse center point
			center = tuple(map(np.float32, tokens[3][1:-1].split(",")))
			# parse corners
			corners = tokens[2][1:-1].split(")(")
			corners = list(map(lambda x: tuple(map(np.float32, x.split(","))), corners))
			# add to dictionary of objects
			objects[int(tokens[1])] = Object(
					tokens[0],
					int(tokens[1]),
					center,
					corners,
					im=cv2.flip(cv2.imread(tokens[0]), 1),)
	return objects
