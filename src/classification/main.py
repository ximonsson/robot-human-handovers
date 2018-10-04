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
DATA_RATIO = 0.8

# learning parameters
learning_rate = 0.001
epochs = 5
batch_size = 10

# network parameters
dropout = 0.5
outputs = 2
train_layers = ["conv5", "fc8"]

# tensorflow variables, x input, y output and dropout
x = tf.placeholder(tf.float32, [batch_size, alexnet.IN_WIDTH, alexnet.IN_HEIGHT, alexnet.IN_DEPTH])
y = tf.placeholder(tf.float32, [None, outputs])
keep_prob = tf.placeholder_with_default(1.0, shape=())

# load dataset
objects = [
		#"ball",
		"bottle",
		"box",
		"can",
		"cutters",
		"glass",
		"hammer",
		"knife",
		"scissors",
		"mug",
		"pen",
		"pitcher",
		"brush",
		"scalpel",
		"screwdriver",
		#"tube",
		]

N = 11
training_objects = objects[:N]
test_objects = objects[N:]
test_data, _ = data.datasets(DATA, test_objects, 1.0, size=50)

# create network
# create the original alexnet model and add a new fully connected layer to output the grasping class
m = alexnet.model(x, keep_prob)
m = tf.nn.softmax(m)
with tf.variable_scope("grasp_class") as scope:
	weights = tf.get_variable('weights', shape=[alexnet.OUT_CLASSES, outputs], trainable=True)
	biases = tf.get_variable('biases', shape=[outputs], trainable=True)
	m = tf.nn.xw_plus_b(m, weights, biases, name=scope.name)

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

with tf.name_scope("testing"):
	classifier = tf.nn.softmax(m) # for testing output
	test = tf.equal(tf.argmax(classifier, 1), tf.argmax(y, 1))
	test_accuracy = tf.reduce_mean(tf.cast(test, tf.float32))


tf.summary.scalar("accuracy", accuracy)
tf.summary.scalar("cross_entropy", loss)

INPUT_DIMENSIONS = [alexnet.IN_WIDTH, alexnet.IN_HEIGHT, alexnet.IN_DEPTH]


with tf.Session() as s:
	# initialize and load weights
	s.run(tf.global_variables_initializer())
	alexnet.load_weights("data/weights/bvlc_alexnet.npy", s, train_layers)

	# split dataset
	training_data, validation_data = data.datasets(DATA, training_objects, DATA_RATIO, size=50)

	# for each epoch fit the model to the training set and
	# calculate accuracy over the validation set
	# after we have run each epoch we test the model on the test set
	for i in range(epochs):
		print("{} Epoch #{}".format(datetime.now(), i+1))

		# train
		print("Training...", end="", flush=True)
		for batch, X, Y in data.batches(training_data, batch_size, INPUT_DIMENSIONS, outputs):
			s.run(train_op, feed_dict={x: X, y: Y, keep_prob: 1.0-dropout})
			print("\rTraining: {}".format(progressbar(batch+1, len(training_data)/batch_size)), flush=True, end="")
		print()

		# validate
		acc = 0
		print("Validating...", end="", flush=True)
		for batch, X, Y in data.batches(validation_data, batch_size, INPUT_DIMENSIONS, outputs):
			b = batch+1
			acc += s.run(accuracy, feed_dict={x: X, y: Y, keep_prob: 1.0})
			acc /= b
			print(
					"\rValidating: {} => Accuracy {:.4f}%".format(
						progressbar(b, len(validation_data)/batch_size),
						acc*100),
					flush=True,
					end="")
		print("\n{} Validation Accuracy: {:.4f}%\n".format(datetime.now(), acc*100))

	# test accuracy
	print("Testing...", end="", flush=True)
	acc = 0
	for batch, X, Y in data.batches(test_data, batch_size, INPUT_DIMENSIONS, outputs):
		b = batch+1
		acc += s.run(test_accuracy, feed_dict={x: X, y: Y, keep_prob: 1.0})
		acc /= b
		print(
				"\rTesting: {} => Accuracy {:.4f}%".format(
					progressbar(b, len(test_data)/batch_size),
					acc*100),
				flush=True,
				end="")
	print()
	print("{} Test Accuracy: {:.4f}%".format(datetime.now(), acc*100))
	print()
