import numpy as np
import tensorflow as tf
import alexnet
import data
import random
from datetime import datetime
from classification.utils import progressbar, print_step


DATA = "data/classification/images/"
DATA_RATIO = 0.8
INPUT_DIMENSIONS = [alexnet.IN_WIDTH, alexnet.IN_HEIGHT, alexnet.IN_DEPTH]

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

# split dataset
training_data, validation_data = data.datasets(DATA, training_objects, DATA_RATIO)
test_data, _ = data.datasets(DATA, test_objects, 1.0)

print("Training on {} images, validating on {} images, and testing on {}".format(
	len(training_data),
	len(validation_data),
	len(test_data)))


# learning parameters
LEARNING_RATE = 0.001
EPOCHS = 5
BATCH_SIZE = 64

# network parameters
DROPOUT = 0.5
OUTPUTS = 2
train_layers = ["fc6", "fc7", "fc8"]

# tensorflow variables, x input, y output and dropout
x = tf.placeholder(tf.float32, [BATCH_SIZE, alexnet.IN_WIDTH, alexnet.IN_HEIGHT, alexnet.IN_DEPTH])
y = tf.placeholder(tf.float32, [None, OUTPUTS])
keep_prob = tf.placeholder_with_default(1.0, shape=())

# create network
# create the original alexnet model and add a new fully connected layer to output the
# grasping class
m = alexnet.model(x, keep_prob, classes=alexnet.OUTPUTS)
#"""
m = tf.nn.relu(m)
with tf.variable_scope("grasp_class") as scope:
	w = tf.get_variable('weights', shape=[alexnet.OUTPUTS, OUTPUTS], trainable=True)
	b = tf.get_variable('biases', shape=[OUTPUTS], trainable=True)
	m = tf.nn.xw_plus_b(m, w, b, name=scope.name)
#"""

# calculate loss
with tf.name_scope("loss"):
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=m, labels=y))
tf.summary.scalar("cross_entropy", loss)

# training operation applying optimizer function
with tf.name_scope("train"):
	# TODO gradient descent?
	train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

# accuracy of the model
with tf.name_scope("accuracy"):
	correct_pred = tf.equal(tf.argmax(m, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar("accuracy", accuracy)

# testing accuracy
#with tf.name_scope("testing"):
	#classifier = tf.nn.softmax(m) # for testing output
	#test = tf.equal(tf.argmax(classifier, 1), tf.argmax(y, 1))
	#test_accuracy = tf.reduce_mean(tf.cast(test, tf.float32))
#tf.summary.scalar("test_accuracy", test_accuracy)


summary_op = tf.summary.merge_all()

with tf.Session() as s:
	# create writers for summary
	train_writer = tf.summary.FileWriter("results/classification/train_6", s.graph)
	validation_writer = tf.summary.FileWriter("results/classification/validation_6")

	# initialize and load weights
	s.run(tf.global_variables_initializer())
	alexnet.load_weights("data/classification/weights/bvlc_alexnet.npy", s, train_layers)

	# for each epoch fit the model to the training set and
	# calculate accuracy over the validation set
	# after we have run each epoch we test the model on the test set

	n_train_batches_per_epoch = int(len(training_data)/BATCH_SIZE)
	n_validation_batches_per_epoch = int(len(validation_data)/BATCH_SIZE)

	step = 0
	for epoch in range(EPOCHS):
		print("{} Epoch #{}".format(datetime.now(), epoch+1))

		# run training operation on each batch and then summarize
		print("Training...", end="", flush=True)
		for batch, X, Y in data.batches(training_data, BATCH_SIZE, INPUT_DIMENSIONS, OUTPUTS):
			s.run(train_op, feed_dict={x: X, y: Y, keep_prob: 1.0-DROPOUT})
			summary = s.run(summary_op, feed_dict={x: X, y: Y, keep_prob: 1.0})
			train_writer.add_summary(summary, step)
			print_step("{:15} {}", "Training:", progressbar(batch+1, n_train_batches_per_epoch))
			step += 1

		# validate and write summary of the accuracy
		acc = 0
		print("\nValidating...", end="", flush=True)
		for batch, X, Y in data.batches(validation_data, BATCH_SIZE, INPUT_DIMENSIONS, OUTPUTS):
			b = batch + 1
			acc += s.run(accuracy, feed_dict={x: X, y: Y, keep_prob: 1.0})
			print_step("{:15} {} => Accuracy {:.4f}%", "Validating:", progressbar(b, n_validation_batches_per_epoch), acc/b*100)
		print("\n{} Validation Accuracy: {:.4f}%\n".format(datetime.now(), acc/b*100))
		summary_val_acc = tf.Summary()
		summary_val_acc.value.add(tag="validation_accuracy", simple_value=acc/b*100)
		validation_writer.add_summary(summary_val_acc, step)

	# test accuracy
	print("Testing...", end="", flush=True)
	acc = 0
	for batch, X, Y in data.batches(test_data, BATCH_SIZE, INPUT_DIMENSIONS, OUTPUTS):
		b = batch+1
		acc += s.run(accuracy, feed_dict={x: X, y: Y, keep_prob: 1.0})
		print_step("{:15} {} => Accuracy {:.4f}%", "Testing:", progressbar(b, int(len(test_data)/BATCH_SIZE)), acc/b*100)
	print("\n{} Test Accuracy: {:.4f}%".format(datetime.now(), acc/b*100))
