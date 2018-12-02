"""
File: results.py
Description: Parse results and output information.
"""
import os

DIR = "results/classification"

def datasets(run, k):
	with open(os.path.join(DIR, run, "training_images.dat")) as f:
		lines = f.readlines()

	sets = []
	while len(lines):
		i = lines.index("\n")
		sets.append([l.strip() for l in lines[:i]])
		lines = lines[i+2:]

	return sum(sets[:k] + sets[k+1:], []), sets[k]


def common_training_images(run):
	pass
