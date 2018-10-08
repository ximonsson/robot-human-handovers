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
epochs = 10
batch_size = 64

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
		"cutters",
		"glass",
		"knife",
		"scissors",
		"brush",
		"scalpel",
		"can",
		"screwdriver",
		"pitcher",
		"hammer",
		"pen",
		"cup",
		#"tube",
		]

N = 12
training_objects = objects[:N]
test_objects = objects[N:]

# create network
# create the original alexnet model and add a new fully connected layer to output the
# grasping class
m = alexnet.model(x, keep_prob, classes=alexnet.OUT_CLASSES)
m = tf.nn.relu(m)
with tf.variable_scope("grasp_class") as scope:
	w = tf.get_variable('weights', shape=[alexnet.OUT_CLASSES, outputs], trainable=True)
	b = tf.get_variable('biases', shape=[outputs], trainable=True)
	m = tf.nn.xw_plus_b(m, w, b, name=scope.name)

# calculate loss
with tf.name_scope("loss"):
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=m, labels=y))

# training operation applying optimizer function
with tf.name_scope("train"):
	# TODO gradient descent?
	train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
tf.summary.scalar("cross_entropy", loss)

# accuracy of the model
with tf.name_scope("accuracy"):
	correct_pred = tf.equal(tf.argmax(m, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar("accuracy", accuracy)

with tf.name_scope("testing"):
	classifier = tf.nn.softmax(m) # for testing output
	test = tf.equal(tf.argmax(classifier, 1), tf.argmax(y, 1))
	test_accuracy = tf.reduce_mean(tf.cast(test, tf.float32))
tf.summary.scalar("test_accuracy", test_accuracy)


merged_summary = tf.summary.merge_all()


INPUT_DIMENSIONS = [alexnet.IN_WIDTH, alexnet.IN_HEIGHT, alexnet.IN_DEPTH]


with tf.Session() as s:
	# split dataset
	training_data, validation_data = data.datasets(DATA, training_objects, DATA_RATIO)
	test_data, _ = data.datasets(DATA, test_objects, 1.0)

	print("Training on {} images, validating on {} images, and testing on {}".format(
		len(training_data),
		len(validation_data),
		len(test_data)))

	# create writers for summary
	train_writer = tf.summary.FileWriter("results/classification/train", s.graph)
	test_writer = tf.summary.FileWriter("results/classification/test")

	# initialize and load weights
	s.run(tf.global_variables_initializer())
	alexnet.load_weights("data/weights/bvlc_alexnet.npy", s, train_layers)

	# for each epoch fit the model to the training set and
	# calculate accuracy over the validation set
	# after we have run each epoch we test the model on the test set
	for i in range(epochs):
		print("{} Epoch #{}".format(datetime.now(), i+1))

		# train
		print("Training...", end="", flush=True)
		n_train_batches = int(len(training_data)/batch_size)
		for batch, X, Y in data.batches(training_data, batch_size, INPUT_DIMENSIONS, outputs):
			# run training operation and calculate loss after
			s.run(train_op, feed_dict={x: X, y: Y, keep_prob: 1.0-dropout})
			l = s.run(loss, feed_dict={x: X, y: Y, keep_prob: 1.0})
			print(
					"\r{:15} {} => Loss {:.4f}".format(
						"Training:",
						progressbar(batch+1, n_train_batches),
						l),
					flush=True,
					end="")
			# summarize
			summary = s.run(merged_summary, feed_dict={x: X, y: Y, keep_prob: 1.0})
			train_writer.add_summary(summary, batch + epoch * n_train_batches))
		print()

		# validate
		acc = 0
		print("Validating...", end="", flush=True)
		for batch, X, Y in data.batches(validation_data, batch_size, INPUT_DIMENSIONS, outputs):
			b = batch + 1
			acc += s.run(accuracy, feed_dict={x: X, y: Y, keep_prob: 1.0})
			print(
					"\r{:15} {} => Accuracy {:.4f}%".format(
						"Validating:",
						progressbar(b, int(len(validation_data)/batch_size)),
						(acc / b * 100)),
					flush=True,
					end="")
		print("\n{} Validation Accuracy: {:.4f}%\n".format(datetime.now(), (acc / (batch + 1) * 100)))

	# test accuracy
	print("Testing...", end="", flush=True)
	acc = 0
	for batch, X, Y in data.batches(test_data, batch_size, INPUT_DIMENSIONS, outputs):
		b = batch+1
		acc += s.run(test_accuracy, feed_dict={x: X, y: Y, keep_prob: 1.0})
		print(
				"\r{:15} {} => Accuracy {:.4f}%".format(
					"Testing:",
					progressbar(b, int(len(test_data)/batch_size)),
					(acc/b)*100),
				flush=True,
				end="")
	print()
	print("{} Test Accuracy: {:.4f}%".format(datetime.now(), (acc/b)*100))
	print()
