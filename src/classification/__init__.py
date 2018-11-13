import pickle


"""
__test_objects__ = [
		"new-bottle",
		"new-can",
		"new-cheeseknife",
		"new-cup",
		"new-fork",
		"new-glass",
		"new-jar",
		"new-knive",
		"new-pliers",
		"new-scissors",
		"new-screwdriver",
		"new-spoon",
		"new-wineglass",
		"new-bottle2",
		]
		"""

__obj_name2id__ = {
		# training objects
		"ball":        None,
		"bottle":      21,
		"box":         16,
		"brush":       22,
		"can":         15,
		"cutters":     17,
		"glass":       20,
		"hammer":       2,
		"knife":       12,
		"cup":         23,
		"pen":          3,
		"pitcher":     19,
		"scalpel":      4,
		"scissors":     5,
		"screwdriver": 14,
		"tube":        18,

		# test objects
		"new-beerglass":   100,
		"new-bottle":      101,
		"new-carrafe":     102,
		"new-cup":         103,
		"new-fork":        104,
		"new-glass":       105,
		"new-knife":       106,
		"new-scissors":    107,
		"new-spatula":     108,
		"new-spoon":       109,
		"new-wineglass":   110,
		"new-woodenspoon": 111,
		}


__CLASS_ASSIGNMENT_FP__ = "data/classification/classes.pkl"

with open(__CLASS_ASSIGNMENT_FP__, "rb") as f:
	__class_assignments__ = pickle.load(f)


class Object:
	def __init__(self, name):
		self.name = name
		self.ID = __obj_name2id__[name]
		for c, objects in __class_assignments__.items():
			if self.ID in objects:
				self.cl = c

	def __str__(self):
		return "[{}] {} ({})".format(self.ID, self.name, self.cl)


__training_objects__ = [
		#"ball",
		"bottle",
		#"box",
		#"cutters",
		"glass",
		"knife",
		"scissors",
		"brush",
		"scalpel",
		"can",
		"screwdriver",
		"pitcher",
		"hammer",
		"pen",
		"cup",
		#"tube",
		]
TRAIN_OBJECTS = [Object(name) for name in __training_objects__]

__test_objects__ = [
		"new-beerglass",
		"new-bottle",
		"new-carrafe",
		"new-cup",
		"new-fork",
		"new-glass",
		"new-knife",
		"new-scissors",
		"new-spatula",
		"new-spoon",
		"new-wineglass",
		"new-woodenspoon",
		]
TEST_OBJECTS = [Object(name) for name in __test_objects__]
