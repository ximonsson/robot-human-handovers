import numpy as np
import tensorflow as tf
import alexnet

# learning parameters
learning_rate = 0.001
epochs = 10
batch_size = 128

# network parameters
dropout = 0.5
outputs = 2
train_layers = ["fc6", "fc7"]

# tensorflow variables, x input, y output and dropout
x = tf.placeholder(tf.float32, [batch_size, alexnet.IN_WIDTH, alexnet.IN_HEIGHT, alexnet.IN_DEPTH])
y = tf.placeholder(tf.float32, [None, outputs])
dropout = tf.placeholder_with_default(1.0, shape=())

# calculate loss
with tf.name_scope("loss"):
    pass

# training operation applying optimizer function
with tf.name_scope("train"):
    pass

# create network
m = alexnet.model(x, dropout, outputs)
softmax = tf.nn.softmax(m)

with tf.Session() as session:
    s.run(tf.global_variables_initializer())
    alexnet.load_weights("weights/bvlc_alexnet.npy", session, train_layers)
