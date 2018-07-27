import alexnet
import tensorflow as tf
import cv2
import numpy as np
import os
#import matplotlib.pyplot as plt
from caffe_classes import class_names
import sys

#%matplotlib inline
image = cv2.imread(sys.argv[1])

# load image
imagenet_mean = np.array([104., 117, 124.], dtype=np.float32)

# setup network
dropout = tf.placeholder_with_default(1.0, shape=())
x = tf.placeholder(tf.float32, [1, alexnet.IN_HEIGHT, alexnet.IN_WIDTH, alexnet.IN_DEPTH])
m = alexnet.model(x, dropout)
softmax = tf.nn.softmax(m)

with tf.Session() as s:
    # initialize and load variables
    s.run(tf.global_variables_initializer())
    alexnet.load_weights("weights/bvlc_alexnet.npy", s, [])

    # reshape the image as to be the correct input size for the network
    img = cv2.resize(image.astype(np.float32), (alexnet.IN_WIDTH, alexnet.IN_HEIGHT))
    img -= imagenet_mean
    img = img.reshape((1, alexnet.IN_HEIGHT, alexnet.IN_WIDTH, alexnet.IN_DEPTH))

    probs = s.run(softmax, feed_dict={x: img})
    c = class_names[np.argmax(probs)]

    print("Class: '%s', probability: %.4f" % (c, probs[0, np.argmax(probs)]))
    #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #plt.title("Class: " + c + ", probability: %.4f" % probs[0,np.argmax(probs)])
    #plt.axis('off')
