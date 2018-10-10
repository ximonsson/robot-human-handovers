import numpy as np
import cv2


__DB_DELIMITER__ = ':'


class Object:
	"""
	Reference object.
	Contains information about filepath to image on disk, center point of the tag, corner
	points and tag ID.
	"""
	def __init__(self, filename, tid, tcenter, corners, center=None, im=None):
		"""
		Create a new object.

		:param filename: string - filepath to the image of the object
		:param tid: integer - AprilTag ID
		:param tcenter: tuple - center point of the tag in the image
		:param corners: array-like - 4 points of the corners of the tag in the image
		:param im: np.array - loaded matrix of the image at filename, preferably by using cv2.imread
		"""
		self.filename = filename
		self.tag_id = tid
		self.tag_center = tcenter
		self.corners = corners
		self.__image__ = im
		self.__center__ = center
		self.__area__ = None
		self.__mask__ = None

	def __str__(self):
		data = [
				self.filename,
				self.tag_id,
				"".join(map(lambda x: "(%s)" % ",".join(map(str, x)), self.corners)),
				"(%s)" % ",".join(map(str, self.tag_center)),
				"(%s)" % ",".join(map(str, self.center)),
				]
		return __DB_DELIMITER__.join(map(str, data))

	@property
	def image(self):
		"""
		Returns the image that is located on disk designated by the filename property
		:returns: np.array - image of the object
		"""
		if self.__image__ is None:
			self.__image__ = cv2.imread(self.filename)
		return self.__image__

	@property
	def mask(self):
		"""
		Returns the mask of the object
		:returns: np.array - binary image
		"""
		if self.__mask__ is None:
			self.__load_properties__()
		return self.__mask__


	def __load_properties__(self):
		# load mask file
		filename = self.filename.replace("%d" % self.tag_id, "%d_mask" % self.tag_id)
		self.__mask__ = cv2.flip(cv2.imread(filename), 1)

		# find (largest) contour around the object
		bw = cv2.cvtColor(self.__mask__, cv2.COLOR_BGRA2GRAY)
		_, contours, _ = cv2.findContours(bw, 1, 2)
		contours = sorted(contours, key=cv2.contourArea)
		cnt = contours[-1]

		# calculate area and center
		self.__area__ = cv2.contourArea(cnt)
		M = cv2.moments(cnt)
		cx = int(M['m10']/M['m00'])
		cy = int(M['m01']/M['m00'])
		self.__center__ = (cx, cy)

	@property
	def center(self):
		"""
		Get the center of mass of the object.
		The mask file is loaded from disk and the largest contour is computed around it.
		Using the contour the center of mass is computed through it's moments.
		:returns: tuple - 2D point
		"""
		if self.__center__ is None:
			self.__load_properties__()
		return self.__center__

	@property
	def area(self):
		"""
		Get the area of the object
		:returns: float
		"""
		if self.__area__ is None:
			self.__load_properties__()
		return self.__area__



def load_objects_database(filepath):
	"""
	Returns dictionary with objects loaded from database file at filepath.

	:params filepath: filepath to object database
	:returns: dictionary with objects with the tag IDs as keys.
	"""
	objects = dict()
	with open("data/objects/objects.db") as f:
		for line in f:
			tokens = line[:-1].split(__DB_DELIMITER__)
			# parse center point
			tag_center = tuple(map(np.float32, tokens[3][1:-1].split(",")))
			center = tuple(map(np.float32, tokens[4][1:-1].split(",")))
			# parse corners
			corners = tokens[2][1:-1].split(")(")
			corners = list(map(lambda x: tuple(map(np.float32, x.split(","))), corners))
			# add to dictionary of objects
			objects[int(tokens[1])] = Object(
					tokens[0],
					int(tokens[1]),
					tag_center,
					corners,
					center=center,
					im=cv2.flip(cv2.imread(tokens[0]), 1),)
	return objects


def store_objects_database(objects, filepath):
	"""
	Store the objects to disk as database.

	:param objects: dict - objects and their data
	:param filepath: string - location on disk to store the database
	"""
	with open(filepath, 'w') as f:
		for o in objects.values():
			f.write(str(o)+"\n")
