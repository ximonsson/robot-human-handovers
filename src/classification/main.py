from sklearn import cluster as skcluster
import numpy as np
import sys

N_FEATURES = 4
X_FEATURE = 1
Y_FEATURE = 3
Z_FEATURE = 4

samples_file = sys.argv[1]
n_clusters = int(sys.argv[2])


def print_summary(k, samples):
	print("*** Summary ***")
	print("%d features and %d samples" % (samples.shape[1]-1, samples.shape[0]))

	print("Cluster assignment:")
	unique, counts = np.unique(k.labels_, return_counts=True)
	d = dict(zip(unique, counts))
	for label in d:
		print(" > #%d => %d" % (label, d[label]))

	print("Object assignments:")
	unique, counts = np.unique(samples[:, 0], return_counts=True)
	d = dict(zip(unique, counts))
	for tid in d:
		print(" > Object ID #%d [ %d ]" % (tid, d[tid]))
		c = dict()
		for i, s in enumerate(samples):
			if s[0] == tid:
				l = k.labels_[i]
				if not l in c:
					c[l] = 0
				c[l] += 1
		for label in c:
			print("    * [ %d ] => %d " % (label, c[label]))


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
	k = skcluster.KMeans(n_clusters=n_clusters)
	k.fit(samples[:, 1:1+N_FEATURES])
	return k


with open(samples_file, "rb") as f:
	samples = np.load(f)
	k = cluster(samples)
	print_summary(k, samples)
	plot(k, samples)
