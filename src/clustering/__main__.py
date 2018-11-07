from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

import numpy as np
import sys
import cv2
import math
import matplotlib.pyplot as plt

from handoverdata.object import load_objects_database


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


# visualize the clustering
FLAG_VISUALIZE = False

def print_summary(cluster_assignments, sample_assignments, k, samples):
	print("*** Summary ***")
	print("Object assignments:")
	for oid in sample_assignments:
		object_samples = sample_assignments[oid]
		object_samples_total = list(samples[:, 0]).count(oid)
		largest_label = max(object_samples, key=object_samples.get)
		print(" > Object #{} [Label #{}: {} samples]".format(
			oid,
			largest_label,
			object_samples_total))
		for label in object_samples:
			print("   => Label #{}: {}% ({} samples)".format(
				label,
				int(object_samples[label]/object_samples_total*100),
				object_samples[label]))
		# visualize the grasp of the cluster on this object
		if FLAG_VISUALIZE:
			display_object(oid, largest_label, k.cluster_centers_[largest_label])
			wait()
		#im = draw(oid, largest_label, k.cluster_centers_[largest_label])
		#cv2.imwrite("results/clustering/{}_handover.jpg".format(oid), im)

	print("Cluster information:")
	for label in cluster_assignments:
		print(" > Label #{} >> Centroid {}".format(
			label,
			k.cluster_centers_[label]))
		print("   Objects: {}".format(cluster_assignments[label]))


def __plot__(k, samples, n_clusters, x=0, y=1, z=2):
	from mpl_toolkits.mplot3d import Axes3D
	color = plt.cm.rainbow(np.linspace(0, 1, n_clusters))
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	for i in range(n_clusters):
		centroid = k.cluster_centers_[i]
		ax.scatter(centroid[x], centroid[y], centroid[z], s=20, c='black')
		s = np.array([samples[j] for j in range(len(k.labels_)) if k.labels_[j] == i])
		ax.scatter(s[:, x], s[:, y], s[:, z], c=color[i])



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


def __store_dat__(filename, *args):
	with open("results/clustering/{}".format(filename), "w") as f:
		lines = []
		for dataset in args:
			for x in dataset:
				if hasattr(x, "__iter__"):
					lines.append("\t" + " ".join(map(str, x)))
				else:
					lines.append("\t" + str(x))
			lines.append(""); lines.append("")
		f.write("\n".join(lines[:-2]))


def __store__(k, cluster_assignments, sample_assignments):
	import pickle
	import os
	DIR = "data/clustering"
	with open(os.path.join(DIR, "centroids.npy"), "wb") as f:
		np.save(f, k.cluster_centers_)
	with open(os.path.join(DIR, "labels.npy"), "wb") as f:
		np.save(f, k.labels_)
	with open(os.path.join(DIR, "clusters.pkl"), "wb") as f:
		pickle.dump(cluster_assignments, f, pickle.HIGHEST_PROTOCOL)


def __print_pca_info__(pca):
	print(" *** PCA FEATURES ***")
	print(pca.components_)
	print(pca.explained_variance_)
	print(pca.explained_variance_ratio_)
	print(pca.singular_values_)
	print(pca.noise_variance_)
	print()


# -----------------------------------------------------------------------------------------------------
# Start main script
#

# Cluster on:
#	- rotation of object
#	- distance ratio between diagonal of object and distance between centers
#	- ratio of object area and grasp area
#	- direction from object center to grasp center X-axis
#	- direction from object center to grasp center Y-axis

FEATURES = [1,4,7,11,12]
PCA_COMPONENTS = 0.9
#PCA_COMPONENTS = len(FEATURES)

# parse command line arguments

samples_file = sys.argv[1]
n_clusters = int(sys.argv[2])
objects = load_objects_database("data/objects/objects.db")

if "--visualize" in sys.argv:
	FLAG_VISUALIZE = True

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
	cs = range(2, n_clusters)
	scores = np.zeros((len(cs), 2))
	silhouette = np.zeros((len(cs), 2))

	for i, n in enumerate(cs):
		# cluster the samples
		k = KMeans(n_clusters=n)
		k.fit(X)

		# compute total variance and average silhouette score
		scores[i] = [n, k.score(X)]
		silhouette[i] = [n, silhouette_score(X, k.labels_)]
		print("Silhouette and cluster score for {} clusters: {:.4f}, {:.4f}".format(n, silhouette[i][1], scores[i][1]))

		silhouette_sample_values = silhouette_samples(X, k.labels_)
		silhouette_sample_values = [sorted(silhouette_sample_values[k.labels_ == c]) for c in range(n)]

		# cluster data for plotting
		#clusters = {l: [] for l in k.labels_}
		#for i, label in enumerate(k.labels_):
			#clusters[label].append(X[i,:])
		clusters = [X[k.labels_ == c] for c in range(n)]

		__store_dat__("silhouette_sample_values_{}.dat".format(n), *silhouette_sample_values)
		__store_dat__("clusters_{}.dat".format(n), *clusters)
		__store_dat__("centroids_{}.dat".format(n), *k.cluster_centers_)

		# create object summaries of the data
		clusters, object_assignments = summarize(k, samples)

		#print_summary(clusters, object_assignments, k, samples)
		#__plot__(k, samples, n)
		#__store__(k, clusters, object_assignments)
		#plt.show()

	# store data for scores and silhouette coefficients
	__store_dat__("scores.dat", scores)
	__store_dat__("silhouette.dat", silhouette)

cv2.destroyAllWindows()
