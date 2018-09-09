import cv2
import numpy as np
import json
from classification.samples import read_samples

with open("data/training/session1/progress.json") as f:
	data = json.load(f)

with open("data/training/session1/raw") as f:
	samples = read_samples(f, data["valid"])

with open("samples.npy", "wb") as f:
	np.save(f, samples)

print(samples)
