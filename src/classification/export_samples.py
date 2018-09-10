import cv2
import numpy as np
import json
import sys
from samples import read_samples

directory = sys.argv[1]

with open("%s/progress.json" % directory) as f:
	data = json.load(f)

with open("%s/raw" % directory) as f:
	samples = read_samples(f, data["valid"])

with open("samples.npy", "wb") as f:
	np.save(f, samples)
