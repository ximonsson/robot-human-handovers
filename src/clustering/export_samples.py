import numpy as np
import json
import sys
from samples import create_samples

#directory = sys.argv[1]
directories = sys.argv[1:-1]
dst = sys.argv[-1]
samples = []

for directory in directories:
	print("{}...".format(directory))
	with open("%s/progress.json" % directory) as f:
		data = json.load(f)
	with open("%s/raw" % directory) as f:
		s = create_samples(f, data["valid"])
	samples.append(s)

samples = np.concatenate(samples)
with open(dst, "wb") as f:
	np.save(f, samples)
