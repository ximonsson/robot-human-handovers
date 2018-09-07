"""
File: samples.py
Description: Functions to retrieve samples used for clustering from handover data.
"""
import numpy as np
from handoverdata.helpers import rotation_matrix
import math


# number of features per sample
FEATURES = 4
OBJECTS_DB_FILE = "data/objects/objects.db"


# load objects into memory
objects = None
if objects is None:
	from handoverdata.object import load_objects_database
	objects = load_objects_database(OBJECTS_DB_FILE)


def handover2sample(h):
	"""
	Converts a Handover object to a sample used for clustering.
	One sample contains:
		- Rotation in z-axis
		- Distance between centers of object and grasp
		- Direction from object center to the grasp center (unit vector of distance)
		- Ratio between object area and grasp area

	:param h: Handover object
	:returns: list of features
	"""
	obj = objects[h.objectID]
	v = np.array(np.array((h.grasp.x, h.grasp.y)) - np.array(obj.center)) # vector between centers

	# distance between centers
	d = np.linalg.norm(v)

	# direction from object center to grasp center
	u = v / d

	# rotation in Z-axis
	R = rotation_matrix(h.H)
	z = math.atan2(R[1,0], R[0,0]) # rotation in Z-axis
	z = -z * 180 / math.pi

	# ratio between object area and grasp area
	ga = h.grasp.w * h.grasp.h
	r = ga / obj.area

	sample = [z, d, u[0], u[1], r]
	return sample


def read_samples(f, indices):
	"""
	Reads handover data with supplied indices from the file pointer and returns them in
	the form of samples to be used.

	:param f: File object -  file pointer to handover data
	:param indices: list - handover indices in the file
	:returns: np.array - samples for clustering
	"""
	from handoverdata.data import read_at
	samples = []
	for i, idx in enumerate(indices):
		samples.append(handover2sample(read_at(f, i)))
	return np.array(samples)
