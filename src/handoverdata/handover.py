import cv2
from handoverdata.helpers import rotation_matrix
import math


class Handover:
	"""
	Handover data.
	Contains information about which file on disk
	"""
	def __init__(self, f, oid, g, h):
		"""
		:params f: string - filepath to frame with the handover
		:params oid: integer - object ID of the object being handed over
		:params g: Grasp object - grasp
		:params h: np.array -
			Homography matrix of the transformation for the object from ground
			zero to as observed in the image.
		"""
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
		im = cv2.warpAffine(im, R, (im.shape[0], im.shape[1]))

		# write text with information about the handover
		im = cv2.putText(
				im,
				"rotation: {} degrees".format(thetaZ),
				(15, 15),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.5,
				(255, 255, 255))

		return im
