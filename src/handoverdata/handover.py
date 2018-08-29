import cv2

class Handover:
	"""
	Handover data.
	Contains information about which file on disk
	"""
	def __init__(self, f, oid, g, h):
		self.filename = f
		self.objectID = oid
		self.grasp = g
		self.H = h
		self.im = cv2.flip(cv2.imread(f), 1)
