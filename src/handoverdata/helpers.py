import cv2
import numpy as np

def rotation_matrix(H):
	"""
	Get the rotation matrix from the homography matrix H

	:params H: homography matrix
	:returns: 3x3 euclidean rotation matrix
	"""
	H = np.copy(H)
	# Normalization to ensure that ||c1|| = 1
	norm = np.sqrt(np.sum(H[:,0] ** 2))
	H /= norm
	c1 = H[:, 0]
	c2 = H[:, 1]
	c3 = np.cross(c1, c2)

	# create rotation matrix
	# calculate the rotation in Z-axis and rotate the original image
	R = np.zeros((3, 3), dtype=np.float64)
	for i in range(3):
		R[i, :] = [c1[i], c2[i], c3[i]]
	w, u, t = cv2.SVDecomp(R)
	return np.dot(u, t)

