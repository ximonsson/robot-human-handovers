import cv2
from handoverdata.helpers import rotation_matrix
import math


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

	def draw(self, item):
		"""
		Draw the handover data on the object.
		The image of the object is rotated accordingly and the grasping region is visualized as a rectangle.
		Returns a new image with the data drawn onto it.
		"""
		# draw the grasping region
		im = self.grasp.draw(item.image)

		# warp it to the same perspective as in the handover
		#item = cv2.warpPerspective(item, handover.H, (item.shape[0], item.shape[1]))
		R = rotation_matrix(self.H)
		thetaZ = math.atan2(R[1,0], R[0,0]) # rotation in Z-axis
		thetaZ = - thetaZ * 180 / math.pi
		R = cv2.getRotationMatrix2D(item.tag_center, thetaZ, 1.0)
		return cv2.warpAffine(im, R, (im.shape[0], im.shape[1]))
