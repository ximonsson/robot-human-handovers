from sklearn import cluster
import numpy as np
import sys

samples_file = sys.argv[1]
n_clusters = int(sys.argv[2])

with open(samples_file, "rb") as f:
	samples = np.load(f)
	k = cluster.KMeans(n_clusters=n_clusters)
	k.fit(samples)
	print(k.labels_)
