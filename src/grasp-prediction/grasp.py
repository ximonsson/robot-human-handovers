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
train_layers = ["fc6", "fc7", "fc8"]

# tensorflow variables, x input, y output and dropout
x = tf.placeholder(tf.float32, [batch_size, alexnet.IN_WIDTH, alexnet.IN_HEIGHT, alexnet.IN_DEPTH])
y = tf.placeholder(tf.float32, [None, outputs])
keep_prob = tf.placeholder_with_default(1.0, shape=())

# create network
m = alexnet.model(x, keep_prob, classes=outputs)

# calculate loss
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=m, labels=y))

# training operation applying optimizer function
with tf.name_scope("train"):
    # TODO gradient descent?
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

correct_pred = tf.equal(tf.argmax(m, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar("accuracy", accuracy)

with tf.Session() as s:
    # initialize and load weights
    s.run(tf.global_variables_initializer())
    alexnet.load_weights("data/weights/bvlc_alexnet.npy", s, train_layers)

    # train
    s.run(train_op, feed_dict={x: None, y: None, keep_prob:1.0-dropout})
