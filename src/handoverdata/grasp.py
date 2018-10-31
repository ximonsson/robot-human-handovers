import numpy as np
import cv2


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

	def __dict__(self):
		return {
			"x": self.x,
			"y": self.y,
			"w": self.w,
			"h": self.h,
			"a": self.a,
			}

	def box(self):
		"""
		Get OpenCV compatible data for the rotated rectangle.
		"""
		return (self.x, self.y), (self.w, self.h), self.a

	@property
	def area(self):
		return float(self.w * self.h)

	def draw(self, im, color=(0, 0, 255), thickness=1):
		"""
		Draw the grasping region on the image.
		Returns a copy of the image with the rectangle drawn on it.
		"""
		im = np.copy(im)
		box = cv2.boxPoints(self.box())
		box = np.int0(box)
		cv2.drawContours(im, [box], 0, color, thickness)
		return im
