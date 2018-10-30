from sklearn import cluster as skcluster
import numpy as np
import sys
import cv2
from handoverdata.object import load_objects_database
import math


N_FEATURES = 7

samples_file = sys.argv[1]
n_clusters = int(sys.argv[2])
objects = load_objects_database("data/objects/objects.db")


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


X_FEATURE = 0
Y_FEATURE = 1
Z_FEATURE = 2

def plot(k, samples, x=X_FEATURE, y=Y_FEATURE, z=Z_FEATURE):
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	color = plt.cm.rainbow(np.linspace(0, 1, n_clusters))
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	for i in range(n_clusters):
		centroid = k.cluster_centers_[i]
		ax.scatter(centroid[x], centroid[y], centroid[z], s=20, c='black')
		s = np.array([samples[j] for j in range(len(k.labels_)) if k.labels_[j] == i])
		ax.scatter(s[:, x], s[:, y], s[:, z], c=color[i])
	plt.show()


def cluster(samples):
	from sklearn.preprocessing import StandardScaler
	from sklearn.decomposition import PCA

	#
	# Cluster on:
	#	- rotation of object
	#	- direction from object center to grasp center
	#	- distance ratio between diagonal of object and distance between centers
	#	- ratio of object area and grasp area
	#	- rotation of the grasp region
	#
	# Perform standard scaling, PCA to get the most variant features
	#

	X = samples[:, [1,2,3,5,6]]
	#X = StandardScaler().fit_transform(X)
	pca = PCA(.9)
	pca.fit(X)
	X = pca.transform(X)

	#print(pca.explained_variance_ratio_)

	k = skcluster.KMeans(init="random", n_clusters=n_clusters)
	k.fit(X)
	plot(k, X)

	return k


def save(k, cluster_assignments, sample_assignments):
	import pickle
	import os
	DIR = "data/clustering"
	with open(os.path.join(DIR, "centroids.npy"), "wb") as f:
		np.save(f, k.cluster_centers_)
	with open(os.path.join(DIR, "labels.npy"), "wb") as f:
		np.save(f, k.labels_)
	with open(os.path.join(DIR, "clusters.pkl"), "wb") as f:
		pickle.dump(cluster_assignments, f, pickle.HIGHEST_PROTOCOL)


if "--visualize" in sys.argv:
	FLAG_VISUALIZE = True

with open(samples_file, "rb") as f:
	samples = np.load(f)
	k = cluster(samples)
	clusters, object_assignments = summarize(k, samples)
	print_summary(clusters, object_assignments, k, samples)
	#plot(k, samples)
	#save(k, clusters, object_assignments)

cv2.destroyAllWindows()
