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
import operator


# Create network
# create the original alexnet model and add a new fully connected layer to output the

# learning parameters
LEARNING_RATE = float(find_arg("learning-rate", "1e-06"))
EPOCHS = int(find_arg("epochs", "20"))
BATCH_SIZE = int(find_arg("batch-size", "32"))
K = int(find_arg("k", "5"))
DROPOUT = float(find_arg("dropout", "0.5"))

# network parameters
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
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=net, labels=y))

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

LOGDIR = "{}/LR-{}__EP-{}__BS-{}__K-{}__D-{}".format(
		"results/classification/", LEARNING_RATE, EPOCHS, BATCH_SIZE, K, DROPOUT)

LOGDIR_SUFFIX = find_arg("logdir-suffix", "")
if LOGDIR_SUFFIX != "":
	LOGDIR = "{}_{}".format(LOGDIR, LOGDIR_SUFFIX)

if not os.path.exists(LOGDIR): # make sure it exists
	os.mkdir(LOGDIR)

VISUALIZATION_STEP = 10

summary_loss = [[] for k in range(K)]
summary_test_acc = [[] for k in range(K)]
summary_val_acc = [[] for k in range(K)]
object_accuracy = [[] for k in range(K)]
confmats = []
bad_images = []

# Prepare data for training, validation and testing.
#	Training and validation sets share objects but have their images split between them.
#	Test data are new objects that are not part of the training phase.

DATA_TRAIN = find_arg("train-data", "data/classification/images-train/")
DATA_TEST = find_arg("test-data", "data/classification/images-test/")
INPUT_DIMENSIONS = [alexnet.IN_WIDTH, alexnet.IN_HEIGHT, alexnet.IN_DEPTH]

# load dataset

# parse arguments for which objects to train on
train_objects_ = find_arg("train-objects", "").split(",")
if len(train_objects_) == 0:
	train_objects = TRAIN_OBJECTS
else:
	train_objects = [o for o in TRAIN_OBJECTS if o.name in train_objects_]

print(train_objects)

# split dataset
training_sets = datasets(DATA_TRAIN, train_objects, K)
test_data = datasets(DATA_TEST, TEST_OBJECTS, 1)[0]

for k in range(K):
	print("*** K={} ***".format(k+1))

	with tf.Session() as s:

		# initialize and load weights
		s.run(tf.global_variables_initializer())
		alexnet.load_weights("data/classification/weights/bvlc_alexnet.npy", s, train_layers)

		# for each epoch fit the model to the training set and
		# calculate accuracy over the validation set
		# after we have run each epoch we test the model on the test set

		step = 0

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
				loss_ = s.run([loss, train_op], feed_dict={x: X, y: Y, keep_prob: 1.0-DROPOUT})[0]
				step += 1
				summary_loss[k].append([step, loss_])
				print_step(
						"Training...   {}, Loss: {:.4f}",
						progressbar(batch+1, n_train_batches_per_epoch),
						loss_)

			# calculate validation accuracy
			print_step(
					"Validating... {}, Loss: {:.4f}",
					progressbar(batch+1, n_train_batches_per_epoch),
					loss_)
			val_acc = 0
			for b, X, Y in batches(validation_data, BATCH_SIZE, INPUT_DIMENSIONS, OUTPUTS):
				val_acc += s.run(accuracy, feed_dict={x: X, y: Y, keep_prob: 1.0})
			val_acc /= (b + 1)
			summary_val_acc[k].append([step, val_acc])

			print_step(
					"Testing...    {}, Loss: {:.4f}, Val: {:.2f}%",
					progressbar(batch+1, n_train_batches_per_epoch),
					loss_,
					val_acc * 100)

			# calculate test accuracy
			test_acc = 0
			for b, X, Y in batches(test_data, BATCH_SIZE, INPUT_DIMENSIONS, OUTPUTS):
				test_acc += s.run(test_accuracy, feed_dict={x: X, y: Y, keep_prob: 1.0})
			test_acc /= (b + 1)
			summary_test_acc[k].append([step, test_acc])

			print_step(
					"Done          {}, Loss: {:.4f}, Val: {:.2f}%, Test: {:.2f}%",
					progressbar(batch+1, n_train_batches_per_epoch),
					loss_,
					val_acc * 100,
					test_acc * 100)
			print()

		# create confusion matrix
		cm = np.zeros((OUTPUTS, OUTPUTS), dtype=np.int32)
		for batch, X, Y in batches(test_data, BATCH_SIZE, INPUT_DIMENSIONS, OUTPUTS):
			cm += s.run(confusion_matrix, feed_dict={x: X, y: Y, keep_prob: 1.0})
		confmats.append(cm)

		# check accuracy of each object
		for o in TEST_OBJECTS:
			test_data_ = [os.path.join(DATA_TEST, f) for f in o.files(DATA_TEST)]
			acc = 0
			bs = 1

			for batch, X, Y, in batches(test_data_, bs, INPUT_DIMENSIONS, OUTPUTS):
				predictions = s.run(test, feed_dict={x: X, y: Y, keep_prob: 1.0})
				acc += s.run(tf.reduce_mean(tf.cast(predictions, tf.float32)))
				bad_images += [test_data_[batch * bs + i] for i, pred in enumerate(predictions) if not pred]

			print(" '{}': {:.2f}%".format(o, acc/(batch+1)*100))
			object_accuracy[k].append(acc/(batch+1)*100)


#
# store data files over the progress
#

with open(os.path.join(LOGDIR, "loss.dat"), "w") as f:
	for summary in summary_loss:
		for v in summary:
			f.write("\t{}\n".format(" ".join(map(str, v))))
		f.write("\n\n")

with open(os.path.join(LOGDIR, "acc_val.dat"), "w") as f:
	for summary in summary_val_acc:
		for v in summary:
			f.write("\t{}\n".format(" ".join(map(str, v))))
		f.write("\n\n")

with open(os.path.join(LOGDIR, "acc_test.dat"), "w") as f:
	for summary in summary_test_acc:
		for v in summary:
			f.write("\t{}\n".format(" ".join(map(str, v))))
		f.write("\n\n")

with open(os.path.join(LOGDIR, "confusion_matrix.dat"), "w") as f:
	for cm in confmats:
		f.write("- {}\n".format(" ".join(map(lambda x: str(x+1), range(OUTPUTS)))))
		for i in range(len(cm)):
			f.write("{} {}\n".format(str(i+1), " ".join(map(str, cm[i]))))
		f.write("\n\n")

with open(os.path.join(LOGDIR, "acc_object.dat"), "w") as f:
	f.write("#\t{}\n".format(" ".join(map(lambda x: x.name, TEST_OBJECTS))))
	for k in range(K):
		f.write("\t{}\n".format(" ".join(map(str, object_accuracy[k]))))
		f.write("\n\n")

with open(os.path.join(LOGDIR, "bad_images.dat"), "w") as f:
	f.writelines(["\t{}\n".format(im) for im in bad_images])

with open(os.path.join(LOGDIR, "training_images.dat"), "w") as f:
	for tset in training_sets:
		f.writelines(["\t{}\n".format(im) for im in tset])
		f.write("\n\n")

with open(os.path.join(LOGDIR, "test_images.dat"), "w") as f:
	f.writelines(["\t{}\n".format(im) for im in test_data])
