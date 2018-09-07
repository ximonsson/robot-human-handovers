import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json


class Cluster:
	"""
	Cluster of samples.
	"""
	def __init__(self, samples, centroid):
		self.samples = samples
		self.centroid = centroid

	def plot(self, color):
		"""
		Plot the cluster.
		This function will not run matplotlib.pyplot.show(). This will need to be done by the caller.
		:param color: what color to plot the samples in.
		"""
		plt.scatter(self.samples[:, 0], self.samples[:, 1], c=color)
		plt.plot(self.centroid[0], self.centroid[1], markersize=35, marker="x", color="k", mew=10)

	def remove_sample(self, i):
		"""
		Remove sample from cluster
		:param i: sample index
		"""
		self.samples = np.delete(self.samples, i)

	def add_sample(self, sample):
		"""
		Add sample to the cluster
		:param sample: Sample to add to the cluster.
		"""
		self.samples = np.append(self.samples, sample)

	def update_centroid(self, centroid):
		"""
		Update with new centroid.
		:param centroid: new centroid
		"""
		self.centroid = centroid


def plot_clusters(clusters):
	"""
	Plot the samples corresponding to the clusters and their centroids.

	:param clusters: clusters to plot
	"""
	color = plt.cm.rainbow(np.linspace(0, 1, len(clusters)))
	for i, c in enumerate(clusters):
		c.plot(color[i])
	plt.show()


def create_centroids(samples, k):
	"""
	Generate k centroids randomly within the samples.

	:param samples: All the samples
	:param k: number of centroids to generate
	:returns: centroids
	"""
	n = tf.shape(samples)[0]
	i = tf.random_shuffle(tf.range(0, n))
	begin = [0,]
	size = [k,]
	size[0] = n
	centroids = tf.slice(i, begin, size)
	centroids = tf.gather(samples, centroids)
	return centroids


def update_centroids(samples, centroids):
	pass


def assign_samples(samples, centroids):
	pass


def load_samples(fp):
	"""
	Load samples from file.

	:param fp: filepath to samples database
	:returns: samples
	"""
	with open(fp) as f:
		return json.load(f)
	return np.array([])



samples = load_samples("")
c = create_centroids(samples)
model = tf.global_variables_initializer()
with tf.Session() as s:
	values = s.run(samples)
	centroids = s.run(c)

plot_clusters(None)
