from sklearn import cluster as skcluster
import numpy as np
import sys
import cv2
from handoverdata.object import load_objects_database
import math


N_FEATURES = 7
X_FEATURE = 0
Y_FEATURE = 1
Z_FEATURE = 2


samples_file = sys.argv[1]
n_clusters = int(sys.argv[2])
objects = load_objects_database("data/objects/objects.db")



def wait():
	k = cv2.waitKey(0)
	while k != ord('q'):
		k = cv2.waitKey(0)


def display_object(oid, label, centroid):
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


def print_summary(k, samples):
	print("*** Summary ***")
	print("%d features and %d samples" % (samples.shape[1]-1, samples.shape[0]))

	print("Object assignments:")
	label_assignments = dict()
	unique, counts = np.unique(samples[:, 0], return_counts=True)
	d = dict(zip(unique, counts))
	for tid in d:
		print(" > Object #%d [%d samples]" % (tid, d[tid]))
		c = dict()
		for i, s in enumerate(samples):
			if s[0] == tid:
				l = k.labels_[i]
				if not l in c:
					c[l] = 0
				c[l] += 1
		largest_label = 0
		label_count = 0
		for label in c:
			if c[label] > label_count:
				largest_label = label
				label_count = c[label]
			print("    * Label #%d => %d (%d%%) " % (label, c[label], c[label]/d[tid]*100))
		if largest_label not in label_assignments:
			label_assignments[largest_label] = []
		label_assignments[largest_label].append(tid)
		display_object(tid, largest_label, k.cluster_centers_[largest_label])
		wait()

	print("Cluster information:")
	unique, counts = np.unique(k.labels_, return_counts=True)
	d = dict(zip(unique, counts))
	for label in d:
		print(" > Label #%d: %d samples >> Centroid %s" % (label, d[label], k.cluster_centers_[label]))
		print("   Objects: {}".format(label_assignments[label]))

	cv2.destroyAllWindows()


def plot(k, samples):
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	color = plt.cm.rainbow(np.linspace(0, 1, n_clusters))
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	for i in range(n_clusters):
		centroid = k.cluster_centers_[i]
		ax.scatter(centroid[X_FEATURE-1], centroid[Y_FEATURE-1], centroid[Z_FEATURE-1], s=20, c='black')
		s = np.array([samples[j] for j in range(len(k.labels_)) if k.labels_[j] == i])
		ax.scatter(s[:, X_FEATURE], s[:, Y_FEATURE], s[:, Z_FEATURE], c=color[i])
	plt.show()


def cluster(samples):
	X = samples[:, [1, 2, 3, 4]]
	k = skcluster.KMeans(init="random", n_clusters=n_clusters)
	k.fit(X)
	return k


def save(k):
	with open("centroids.npy", "wb") as f:
		np.save(f, k.cluster_centers_)
	#with open("labels.npy", "wb") as f:
		#np.save(f, k.labels_)


with open(samples_file, "rb") as f:
	samples = np.load(f)
	k = cluster(samples)
	print_summary(k, samples)
	plot(k, samples)
	save(k)
