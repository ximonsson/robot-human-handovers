"""
File: samples.py
Description: Functions to retrieve samples used for clustering from handover data.
"""
import numpy as np
import handoverdata as hd
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
		- Object Tag ID
		- Rotation of object in z-axis
		- Direction from object center to the grasp center expressed as an angle in degrees
		- Distance between centers of object and grasp
		- Ratio between object area and grasp area
		- Rotation of the grasp in z-axis
		- Width of grasp
		- Height of grasp

	:param h: Handover object
	:returns: list of features
	"""
	obj = objects[h.objectID]
	v = np.array((h.grasp.x, h.grasp.y)) - np.array(obj.center) # vector between centers

	# distance between centers
	d = np.linalg.norm(v)

	# direction from object center to grasp center
	u = v / d
	u = math.atan(u[0]/u[1])
	u = math.degrees(u)

	# rotation in Z-axis
	R = rotation_matrix(h.H)
	z = math.atan2(R[1,0], R[0,0]) # rotation in Z-axis
	z = math.degrees(z)

	# ratio between object area and grasp area
	ga = h.grasp.w * h.grasp.h
	r = ga / obj.area

	sample = [h.objectID, z, u, d, r, h.grasp.a, h.grasp.w, h.grasp.h]
	return sample


def sample2handover(sample):
	g = hd.Grasp(0, 0, sample[6], sample[7], sample[5])
	h = hd.Handover("", sample[0], g, np.array())
	return h


def read_samples(f, indices):
	"""
	Reads handover data with supplied indices from the file pointer and returns them in
	the form of samples to be used.

	:param f: File object -  file pointer to handover data
	:param indices: array like - handover indices in the file
	:returns: np.array - samples for clustering
	"""
	from handoverdata.data import read_at
	samples = []
	for i, idx in enumerate(indices):
		samples.append(handover2sample(read_at(f, i)))
	return np.array(samples)
