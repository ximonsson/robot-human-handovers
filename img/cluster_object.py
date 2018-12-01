"""
Visualize cluster features on objects.
Iterates over all the objects and applies the handover settings according to their respective cluster
and stores copies to disk.
"""
from clustering.clusters import load_clusters
from handoverdata.object import load_objects_database
from handoverdata import OBJID_2_NAME
import pickle
import cv2
import os


OUTPUTDIR = "img/results/objects/"


def store(cluster, obj):
	im = cluster.apply(obj)
	cv2.imwrite(os.path.join(OUTPUTDIR, "{}.jpg".format(OBJID_2_NAME[obj.tag_id])), im)


with open("results/clustering/cluster-samples_6.dat") as f:
	clusters = load_clusters(f)

with open("results/clustering/object-cluster-assignments_6.pkl", "rb") as f:
	oca = pickle.load(f)

objects = load_objects_database("data/objects/objects.db")

for c, oIDs in oca.items():
	for oID in oIDs:
		store(clusters[c], objects[oID])
