from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

import numpy as np
import sys
import cv2
import math
import matplotlib.pyplot as plt
import pickle
import os

from handoverdata.object import load_objects_database
from handoverdata import OBJID_2_NAME


def wait():
	k = cv2.waitKey(0)
	while k != ord('q'):
		k = cv2.waitKey(0)


def draw(oid, label, centroid):
	obj = objects[oid]
	im = np.copy(obj.image)
	oc = obj.center

	# draw center of grasp
	a = math.radians(centroid[1])
	gc = (oc[0] - centroid[2] * math.sin(a), oc[1] - centroid[2] * math.cos(a))
	gc = tuple(map(int, gc))
	im = cv2.line(im, oc, gc, (255, 0, 0), 2)

	# draw grasping region
	ga = obj.area * centroid[3]
	r = int(np.sqrt(ga/math.pi))
	im = cv2.circle(im, gc, r, (0, 0, 255), 1)

	# rotate
	R = cv2.getRotationMatrix2D(oc, -centroid[0], 1.0)
	im = cv2.warpAffine(im, R, (im.shape[0], im.shape[1]))
	return im


def display_object(oid, label, centroid):
	im = draw(oid, label, centroid)
	# write text
	im = cv2.putText(
			im,
			"object: %d, label: %d, rotation: %d degrees" % (oid, label, centroid[0]),
			(20, 40),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.5,
			(255, 255, 255))
	im = cv2.putText(
			im,
			"grasp center: %s, distance: %d, angle: %d" % (gc, centroid[2], centroid[1]),
			(20, 60),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.5,
			(255, 255, 255))
	# display
	cv2.imshow("opencv object centroid", im)


# -----------------------------------------------------------------------------------------------------
# Here starts code that is used
# The above will most likely be removed later


def summarize(k, samples):
	"""
	Returns summary for the clustering in form of which objects were assigned to which cluster and how
	many samples of each object were assigned to which cluster.
	"""
	cluster_assignments = dict()
	sample_assignments = dict()

	unique, counts = np.unique(samples[:, 0], return_counts=True)
	sample_objects = dict(zip(unique, counts))
	for oid in sample_objects:
		# count the number of samples that belonged to each cluster for this object
		sample_assignments[oid] = dict()
		for i, s in enumerate(samples):
			if s[0] == oid:
				l = k.labels_[i]
				if not l in sample_assignments[oid]:
					sample_assignments[oid][l] = 0
				sample_assignments[oid][l] += 1
		largest_label = max(sample_assignments[oid], key=sample_assignments[oid].get)
		# add the object to the cluster which it has most belonging samples to
		if largest_label not in cluster_assignments:
			cluster_assignments[largest_label] = []
		cluster_assignments[largest_label].append(oid)

	return cluster_assignments, sample_assignments


# directory where to save the results
DIR = "results/clustering"

def store_dat(filename, *args):
	with open(os.path.join(DIR, filename), "w") as f:
		lines = []
		for dataset in args:
			for x in dataset:
				if hasattr(x, "__iter__"):
					lines.append("\t" + " ".join(map(str, x)))
				else:
					lines.append("\t" + str(x))
			lines.append(""); lines.append("")
		f.write("\n".join(lines[:-2]))


def __print_pca_info__(pca):
	print(" *** PCA FEATURES ***")
	print(pca.components_)
	print(pca.explained_variance_)
	print(pca.explained_variance_ratio_)
	print(pca.singular_values_)
	print(pca.noise_variance_)
	print()


def sample_heat_map(object_samples_assignemnts, n):
	m = np.zeros((len(object_samples_assignemnts), n+1), dtype=np.int)
	count = 0
	for oID, assignments in object_samples_assignemnts.items():
		m[count][0] = oID
		for l, c in assignments.items():
			m[count][l+1] = c
		count += 1
	return m


# -----------------------------------------------------------------------------------------------------
# Start main script
#

#
# Cluster on:
#	- rotation of object
#	- distance ratio between diagonal of object and distance between centers
#	- ratio of object area and grasp area
#	- direction from object center to grasp center X-axis
#	- direction from object center to grasp center Y-axis
#
# Perform PCA to project the data on to 3 components
#

FEATURES = [1,4,7,11,12]
PCA_COMPONENTS = 3

# parse command line arguments

samples_file = sys.argv[1]
n_clusters = int(sys.argv[2])
objects = load_objects_database("data/objects/objects.db")

with open(samples_file, "rb") as f:
	#
	# Load samples, extract wanted features from the samples.
	# Scale the features.
	# PCA to get the most variant features.
	# Transform the data to fit the scaling and PCA.
	#
	samples = np.load(f)
	X = samples[:, FEATURES]
	X = StandardScaler().fit_transform(X)
	pca = PCA(PCA_COMPONENTS)
	X = pca.fit_transform(X)
	__print_pca_info__(pca)

	# compute variance per numbers of clusters
	cs = range(2, n_clusters+1)
	scores = np.zeros((len(cs), 2))
	silhouette = np.zeros((len(cs), 2))

	for i, n in enumerate(cs):
		# cluster the samples
		k = KMeans(n_clusters=n)
		k.fit(X)

		# compute total variance and average silhouette score
		scores[i] = [n, k.inertia_]
		silhouette[i] = [n, silhouette_score(X, k.labels_)]
		silhouette_sample_values = silhouette_samples(X, k.labels_)
		silhouette_sample_values = \
				[sorted(silhouette_sample_values[k.labels_ == c], reverse=True) for c in range(n)]

		# sample cluster data for plotting
		clusters = [X[k.labels_ == c] for c in range(n)]
		cluster_samples = [samples[:, FEATURES][k.labels_ == c] for c in range(n)]

		# create object summaries of the data
		cluster_assignments, sample_assignments = summarize(k, samples)

		# store a heat map of the distribution of the original samples into the clusters per object
		hm = sample_heat_map(sample_assignments, n)
		with open(os.path.join(DIR, "object-sample-assignments_{}.dat".format(n)), "w") as f:
			f.write("- {}\n".format(" ".join(map(str, range(1, n+1)))))
			for row in hm:
				f.write("{} {}\n".format(OBJID_2_NAME[row[0]], " ".join(map(str, row[1:]))))

		# store results
		store_dat("silhouette_sample_values_{}.dat".format(n), *silhouette_sample_values)
		store_dat("clusters_{}.dat".format(n), *clusters)
		store_dat("cluster-samples_{}.dat".format(n), *cluster_samples)
		#store_dat("object-sample-assignments_6.dat", hm)
		store_dat(
				"centroids_{}.dat".format(n),
				*k.cluster_centers_.reshape((k.cluster_centers_.shape[0], 1, k.cluster_centers_.shape[1])))
		with open(os.path.join(DIR, "object-cluster-assignments_{}.pkl".format(n)), "wb") as f:
			pickle.dump(cluster_assignments, f)
		with open(os.path.join(DIR, "object-sample-assignments_{}.pkl".format(n)), "wb") as f:
			pickle.dump(sample_assignments, f)

		print(
				"Silhouette and cluster score for {} clusters: {:.4f}, {:.4f}".format(
					n,
					silhouette[i][1],
					scores[i][1]))

	# store data for scores and silhouette coefficients
	store_dat("scores.dat", scores)
	store_dat("silhouette.dat", silhouette)

cv2.destroyAllWindows()
