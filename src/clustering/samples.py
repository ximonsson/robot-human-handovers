"""
File: samples.py
Description: Functions to retrieve samples used for clustering from handover data.
"""
import numpy as np
import handoverdata as hd
from handoverdata.helpers import rotation_matrix
import math


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
		0. Object Tag ID
		1. Rotation of object in z-axis
		2. Direction from object center to the grasp center expressed as an angle in degrees
		3. Distance between centers of object and grasp
		4. Ratio between distance between centers and diagonal of object
		5. Object area
		6. Grasp area
		7. Ratio between object area and grasp area
		8. Rotation of the grasp in z-axis
		9. Width of grasp
		10. Height of grasp

	:param h: Handover object
	:returns: list of features
	"""
	obj = objects[h.objectID]
	v = np.array(h.grasp.center) - np.array(obj.center) # vector between centers

	# distance between centers
	d = np.linalg.norm(v)

	# ratio of distance between center to diagonal of the object
	dr = d / obj.diagonal

	# direction from object center to grasp center (angle)
	u = math.atan(v[0]/v[1])
	if v[1] < 0:
		u += math.pi / 2
	u /= 2 * math.pi

	# rotation of object in Z-axis
	R = rotation_matrix(h.H)
	z = math.atan2(R[1,0], R[0,0]) # rotation in Z-axis
	z /= math.pi

	# ratio between object area and grasp area
	ar = h.grasp.area / obj.area

	sample = [
			h.objectID,
			z,
			u,
			d,
			dr,
			obj.area,
			h.grasp.area,
			ar,
			h.grasp.a,
			h.grasp.w,
			h.grasp.h]
	return sample


def sample2handover(sample):
	"""
	Convert sample to handover object.
	TODO - implement if possible

	:param sample: list -
		sample of handover features in same format as obtained through handover2sample
	:returns: handover.Handover object
	"""
	return None


def create_samples(f, indices):
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
