"""
File: __main__.py
Description:
		Run classification training, validation and testing.

		Takes a number of objects and splits their images between training and validation.
		The remaining objects are used for testing to see at which accuracy the network
		can predict handover class for foreign objects.

		Training loss and accuracy, and validation accuracy are exported for inspection through
		tensorboard.

		Valid command line options are:
			- batch-size: integer
			- learning-rate: float
			- epochs: integer
			- data: string - filepath to directory containing data
			- logdir-suffix: string -
				suffix to add to the log directory to differentiate different runs
"""
import os
import numpy as np
import tensorflow as tf
import random
from datetime import datetime
import classification.alexnet as alexnet
from .data import datasets, batches
from .utils import progressbar, print_step, find_arg
from classification import TEST_OBJECTS, TRAIN_OBJECTS


# Create network
# create the original alexnet model and add a new fully connected layer to output the

# learning parameters
LEARNING_RATE = float(find_arg("learning-rate", "0.001"))
EPOCHS = int(find_arg("epochs", "10"))
BATCH_SIZE = int(find_arg("batch-size", "32"))
K = int(find_arg("k", "5"))

# network parameters
DROPOUT = 0.5
OUTPUTS = 2
train_layers = ["fc6", "fc7", "fc8"]

# tensorflow variables, x input, y output and dropout
x = tf.placeholder(tf.float32, [None, alexnet.IN_WIDTH, alexnet.IN_HEIGHT, alexnet.IN_DEPTH])
y = tf.placeholder(tf.float32, [None, OUTPUTS])
keep_prob = tf.placeholder_with_default(1.0, shape=())

# create network
net = alexnet.network(x, keep_prob, classes=alexnet.OUTPUTS)
net = tf.nn.relu(net)
with tf.variable_scope("grasp_class") as scope:
	w = tf.get_variable('weights', shape=[alexnet.OUTPUTS, OUTPUTS], trainable=True)
	b = tf.get_variable('biases', shape=[OUTPUTS], trainable=True)
	net = tf.nn.xw_plus_b(net, w, b, name=scope.name)

# calculate loss
with tf.name_scope("loss"):
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y))

# training operation applying optimizer function
with tf.name_scope("train"):
	# TODO gradient descent?
	train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

# accuracy of the model
with tf.name_scope("accuracy"):
	correct_pred = tf.equal(tf.argmax(net, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# testing accuracy
with tf.name_scope("testing"):
	classifier = tf.nn.softmax(net) # for testing output
	test = tf.equal(tf.argmax(classifier, 1), tf.argmax(y, 1))
	test_accuracy = tf.reduce_mean(tf.cast(test, tf.float32))
	confusion_matrix = tf.confusion_matrix(tf.argmax(y, 1), tf.argmax(classifier, 1), num_classes=OUTPUTS)

# Setup logging

LOGDIR = "results/classification/"
LOGDIR_SUFFIX = find_arg("logdir-suffix", "") # TODO use this

summary_loss = []
summary_train_acc = [] # TODO
summary_test_acc = []
summary_val_acc = []

# Prepare data for training, validation and testing.
#	Training and validation sets share objects but have their images split between them.
#	Test data are new objects that are not part of the training phase.

DATA_TRAIN = find_arg("train-data", "data/classification/images-train/")
DATA_TEST = find_arg("test-data", "data/classification/images-test/")
INPUT_DIMENSIONS = [alexnet.IN_WIDTH, alexnet.IN_HEIGHT, alexnet.IN_DEPTH]

# load dataset

# split dataset
training_sets = datasets(DATA_TRAIN, TRAIN_OBJECTS, K)
test_data = datasets(DATA_TEST, TEST_OBJECTS, 1)[0]


with tf.Session() as s:
	# initialize and load weights
	s.run(tf.global_variables_initializer())
	alexnet.load_weights("data/classification/weights/bvlc_alexnet.npy", s, train_layers)

	# for each epoch fit the model to the training set and
	# calculate accuracy over the validation set
	# after we have run each epoch we test the model on the test set

	step = 0

	for k in range(K):
		print("*** K={} ***".format(k+1))
		training_data = sum(training_sets[:k] + training_sets[k+1:], [])
		validation_data = training_sets[k]

		n_train_batches_per_epoch = int(len(training_data)/BATCH_SIZE)
		n_validation_batches_per_epoch = int(len(validation_data)/BATCH_SIZE)

		for epoch in range(EPOCHS):
			print("{} Epoch #{}".format(datetime.now(), epoch+1))

			# shuffle the data
			random.shuffle(training_data)

			# run training operation on each batch and then summarize
			print("Training...", end="", flush=True)
			for batch, X, Y in batches(training_data, BATCH_SIZE, INPUT_DIMENSIONS, OUTPUTS):
				s.run(train_op, feed_dict={x: X, y: Y, keep_prob: 1.0-DROPOUT})
				loss_ = s.run(loss, feed_dict={x: X, y: Y, keep_prob: 1.0})
				step += 1

				# validate and write summary of the accuracy
				acc = 0
				#print("\nValidating...", end="", flush=True)
				for batch, X, Y in batches(validation_data, BATCH_SIZE, INPUT_DIMENSIONS, OUTPUTS):
					b = batch + 1
					acc += s.run(accuracy, feed_dict={x: X, y: Y, keep_prob: 1.0})
					#print_step(
							#"{:15} {} => Accuracy {:.4f}%",
							#"Validating:",
							#progressbar(b, n_validation_batches_per_epoch),
							#acc/b*100)

				summary_loss.append(loss_)
				summary_val_acc.append(acc/b*100)
				print_step(
						"{:15} {} Loss: {:.4f}, Val accuracy: {:4.f}",
						"Training:",
						progressbar(batch+1, n_train_batches_per_epoch),
						loss_,
						acc/b*100)

			# check test accuracy
			print("\nTesting...", end="", flush=True)
			acc = 0
			for batch, X, Y in batches(test_data, BATCH_SIZE, INPUT_DIMENSIONS, OUTPUTS):
				b = batch+1
				acc += s.run(test_accuracy, feed_dict={x: X, y: Y, keep_prob: 1.0})
				print_step(
						"{:15} {} => Accuracy {:.4f}%",
						"Testing:",
						progressbar(b, int(len(test_data)/BATCH_SIZE)),
						acc/b*100)
			print()
			summary_test_acc.append(acc/b*100)

	# create confusion matrix
	cm = np.zeros((OUTPUTS, OUTPUTS), dtype=np.int32)
	for batch, X, Y in batches(test_data, BATCH_SIZE, INPUT_DIMENSIONS, OUTPUTS):
		cm += s.run(confusion_matrix, feed_dict={x: X, y: Y, keep_prob: 1.0})

	# check accuracy of each object
	object_accuracy = [[o.name for o in TEST_OBJECTS], []]
	for o in TEST_OBJECTS:
		test_data = datasets(DATA_TEST, [o], 1)[0]
		acc = 0
		for batch, X, Y, in batches(test_data, BATCH_SIZE, INPUT_DIMENSIONS, OUTPUTS):
			acc += s.run(test_accuracy, feed_dict={x: X, y: Y, keep_prob: 1.0})
		print(" '{}': {:.4f}%".format(o, acc/(batch+1)*100))
		object_accuracy[1].append(acc/(batch+1)*100)


# store data files over the progress

with open(os.path.join(LOGDIR, "loss.dat"), "w") as f:
	for v in summary_loss:
		f.write("\t{}\n".format(v))

with open(os.path.join(LOGDIR, "acc_val.dat"), "w") as f:
	for v in summary_val_acc:
		f.write("\t{}\n".format(v))

with open(os.path.join(LOGDIR, "acc_test.dat"), "w") as f:
	for v in summary_test_acc:
		f.write("\t{}\n".format(v))

with open(os.path.join(LOGDIR, "confusion_matrix.dat"), "w") as f:
	for row in cm:
		f.write("\t{}\n".format(" ".join(row)))

with open(os.path.join(LOGDIR, "acc_object.dat"), "w") as f:
	f.write("\t{}\n".format(" ".join(object_accuracy[0])))
	f.write("\t{}\n".format(" ".join(map(str, object_accuracy[1]))))
