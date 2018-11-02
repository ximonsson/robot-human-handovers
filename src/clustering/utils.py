import numpy as np

def object_feature_summary(samples, oid, feature):
	object_samples = samples[np.where(samples[:, 0] == oid)]
	f = object_samples[:, feature]
	return "{:.3f} | {:.3f} | {:.3f} | {:.3f}".format(np.min(f), np.max(f), np.mean(f), np.std(f))

def feature_summary(samples, objects, feature):
	s = ""
	for o in objects:
		s += "{:2d} | {}\n".format(o, object_feature_summary(samples, o, feature))
	return s


