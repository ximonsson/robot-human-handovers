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

	def box(self):
		return (self.x, self.y), (self.w, self.h), self.a

	def __dict__(self):
		return {
			"x": self.x,
			"y": self.y,
			"w": self.w,
			"h": self.h,
			"a": self.a,
			}

