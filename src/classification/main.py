import numpy as np
import tensorflow as tf
import alexnet
import data
import random
from datetime import datetime

def progressbar(done, total):
	progress = done / total
	bar = ""
	bar_len = 30
	for _ in range(int(bar_len * progress)):
		bar += "#"
	for _ in range(bar_len-int(bar_len * progress)):
		bar += "-"
	return "[{}] {}% Done".format(bar, int(progress * 100))


DATA = "data/classification/images/"
DATA_RATIO = 0.7

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

# load dataset
objects = [
		#"ball",
		"bottle",
		"box",
		"brush",
		"can",
		"cutters",
		"glass",
		"hammer",
		"knife",
		"mug",
		"pen",
		"pitcher",
		"scalpel",
		"scissors",
		"screwdriver",
		#"tube",
		]
training_data, validation_data = data.datasets(DATA, objects, DATA_RATIO)
training_data = training_data[:batch_size*10]
validation_data = validation_data[:batch_size*2]

# create network
m = alexnet.model(x, keep_prob, classes=outputs)

# calculate loss
with tf.name_scope("loss"):
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=m, labels=y))

# training operation applying optimizer function
with tf.name_scope("train"):
	# TODO gradient descent?
	train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# accuracy of the model
with tf.name_scope("accuracy"):
	correct_pred = tf.equal(tf.argmax(m, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.summary.scalar("accuracy", accuracy)
tf.summary.scalar("cross_entropy", loss)

with tf.Session() as s:
	# initialize and load weights
	s.run(tf.global_variables_initializer())
	alexnet.load_weights("data/weights/bvlc_alexnet.npy", s, train_layers)

	# for each epoch fit the model to the training set and
	# calculate accuracy over the validation set
	for i in range(epochs):
		print("{} Epoch #{}".format(datetime.now(), i+1))
		batch = 0
		for X, Y in data.batches(training_data, batch_size, [alexnet.IN_WIDTH, alexnet.IN_HEIGHT, alexnet.IN_DEPTH], outputs):
			s.run(train_op, feed_dict={x: X, y: Y, keep_prob: 1.0-dropout})
			batch += 1
			print("\rTraining: {}".format(progressbar(batch, len(training_data)/batch_size)), flush=True, end="")

		print()
		batch = 0
		acc = 0
		for X, Y in data.batches(validation_data, batch_size, [alexnet.IN_WIDTH, alexnet.IN_HEIGHT, alexnet.IN_DEPTH], outputs):
			acc += s.run(accuracy, feed_dict={x: X, y: Y, keep_prob: 1.0})
			batch += 1
			print(
					"\rValidating: {} => Accuracy {:.4f}%".format(
						progressbar(
							batch,
							len(validation_data)/batch_size),
						acc/batch*100),
					flush=True,
					end="")

		acc /= batch
		print()
		print("{} Accuracy: {:.4f}%".format(datetime.now(), acc))
		print()
