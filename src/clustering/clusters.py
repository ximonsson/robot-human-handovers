"""
File: clusters.py
Description:
		file with functions concerning visualization and processing of results
		from clustering.
"""

from handoverdata import Object
import numpy as np
import math
import cv2


class Cluster:
	def __init__(self, samples):
		self.samples = samples

	def mean(self):
		"""
		Return the mean value of every feature in the cluster
		:returns: np.array
		"""
		return np.mean(self.samples, axis=0)

	def apply(self, obj):
		"""
		Visualize the cluster mean features on the object.
		:param obj: handoverdata.Object - object to visualize the settings on
		:returns: np.array -
			Image with the object transformed and so on according to
			the cluster settings.
		"""
		features = self.mean()

		# the features are in order:
		#	- rotation of the object
		#	- ratio between distance of centers and diagonal of the object
		#	- ratio between object and grasp area
		#	- direction in x from the center of the object to center of grasp
		#	- direction in y from the center of the object to center of grasp

		# rotation of the object
		ro = math.degrees(features[0] * math.pi)

		# center of the grasp
		cg = obj.center + np.array([features[3], features[4]]) * obj.diagonal * features[1]
		cg = tuple(cg.astype(np.int))

		# radius of the circle that has the same area as the grasp
		ag = obj.area * features[2]
		radiusg = int(math.sqrt(ag / math.pi))

		# draw everything onto the object image
		im = np.zeros(obj.image.shape, dtype=np.uint8)
		im = cv2.circle(im, cg, radiusg, (100, 255, 0), thickness=cv2.FILLED)
		im = cv2.addWeighted(im, 0.5, obj.image, 0.8, 0)
		im = cv2.drawMarker(im, cg, (0, 0, 255), thickness=2, markerSize=15)
		R = cv2.getRotationMatrix2D(obj.tag_center, -ro, 1.0)
		im = cv2.warpAffine(im, R, (im.shape[0], im.shape[1]))
		return im


def load_clusters(f):
	"""
	Load clusters from stream.
	Will read to EOF of stream pointer (file) and return the clusters found.
	"""
	samples = []
	clusters = []
	for line in f:
		if line == "\n":
			clusters.append(Cluster(samples))
			samples = []
			f.readline()
			continue
		samples.append([float(x) for x in line.strip().split(" ")])
	clusters.append(Cluster(samples))
	return clusters


def visualize(cluster, obj):
	"""
	Apply cluster mean features to the object and visualize it to the screen.
	Press 'q' to exit the visualizing.

	:param cluster: Cluster
	:param obj: handoverdata.Object
	"""
	im = cluster.apply(obj)
	cv2.imshow("opencv", im)
	while True:
		k = cv2.waitKey(0)
		if k == ord('q'):
			break
