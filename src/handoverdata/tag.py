import numpy as np
import cv2


class TagDetection:
	"""
	Detection of an AprilTag within an image
	"""

	def __init__(self, tid, points, center):
		self.id = tid
		self.points = points
		self.center = center

	def __str__(self):
		return "[{}] {}, {}".format(self.id, self.center, self.points)

	def draw(self, image):
		"""
		Draw the tag detection in the image (that it was originally detected within).

		:param image: np.array - image where the tag was originally detected
		:returns: np.array - copy of the image with the detection visualized
		"""
		im = np.copy(image)

		# rectangle around detection
		im = cv2.line(im, self.points[0], self.points[1], (0, 0xff, 0), 2)
		im = cv2.line(im, self.points[1], self.points[2], (0, 0, 0xff), 2)
		im = cv2.line(im, self.points[2], self.points[3], (0xff, 0, 0), 2)
		im = cv2.line(im, self.points[3], self.points[0], (0xff, 0, 0), 2)

		fontface = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
		textsize, _ = cv2.getTextSize (str(self.id), fontface, 1.0, 2)
		center = (
				int(self.center[0] - textsize[0] / 2),
				int(self.center[1] + textsize[1] / 2))
		# text with ID in the center of detection
		im = cv2.putText(im, str(self.id), center, fontface, 1.0, (0xff, 0x99, 0), 2)

		return im
