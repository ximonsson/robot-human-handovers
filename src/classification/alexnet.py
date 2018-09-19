'''
Implementation of the AlexNet model taken from:
    https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html
    http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
'''

import tensorflow as tf
import numpy as np


IN_WIDTH  = 227
IN_HEIGHT = 227
IN_DEPTH  = 3


def __conv__(x, fh, fw, co, sy, sx, name, padding="VALID", group=1):
    """
    __conv__ creates a new 2D convolution layer with tensorflow.
    note that there is no activation function linked to it.

    :param x: kernel input to the layer
    :param fh: filter height
    :param fw: filter width
    :param co: output size
    :param sy: stride y-axis
    :param sx: stride x-axis
    :param name: name of the layer
    :param padding: type of padding
    :param group: AlexNet is divided in two groups, group defines which one this layer belongs to
    :returns: tensor representing the layer
    """
    # get input size
    ci = int(x.get_shape()[-1])
    # convenience function to create a 2d convolutional layer
    convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, sx, sy, 1], padding=padding)

    with tf.variable_scope(name) as scope:
        # variables for weights and biases
        weights = tf.get_variable("weights", shape=[fh, fw, ci/group, co])
        biases = tf.get_variable("biases", shape=[co])

        if group == 1:
            conv = convolve(x, weights)
        else:
            # in case we have mutliple groups we split the inputs and weights
            groups_in = tf.split(x, group, 3)
            groups_weights = tf.split(weights, group, 3)
            groups_out = [convolve(i, k) for i, k in zip(groups_in, groups_weights)]
            # concatenate the output back into one output
            conv = tf.concat(groups_out, 3)

        # add bias and return
        return tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())


def __fc__(x, ci, co, name, relu=True):
    """
    __fc__ returns a new fully connected layer.

    :param x: input kernel to the layer
    :param ci: number of input variables
    :param co: number of output variables
    :param name: name of the layer
    :param relu: add relu activation to the layer if True.
    :returns: tensor representing the layer
    """
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[ci, co], trainable=True)
        biases = tf.get_variable('biases', [co], trainable=True)
        activation = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if relu: # apply relu
            activation = tf.nn.relu(activation)

        return activation


def model(x, dropout, classes=1000):
    """
    model creates a new model with all the layers as defined by AlexNet.
    See https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
    for more documenation on the actual model.

    :param x: tensorflow placeholder for input
    :param dropout: tensor with keep probability for dropout after layers 6 and 7
    :param classes: number of output classes
    :returns: tensor to the last layer
    """

    # 1st layer
    conv1 = __conv__(x, 11, 11, 96, 4, 4, "conv1", padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1, name="conv1")
    lrn1 = tf.nn.local_response_normalization(
            conv1,
            depth_radius=2,
            alpha=2e-05,
            beta=0.75,
            bias=1.0,
            name="norm1")
    pool1 = tf.nn.max_pool(
            lrn1,
            ksize=[1, 3, 3, 1],
            strides=[1, 2, 2, 1],
            padding="VALID",
            name="pool1")

    # 2nd layer
    conv2 = __conv__(pool1, 5, 5, 256, 1, 1, "conv2", padding="SAME", group=2)
    conv2 = tf.nn.relu(conv2, name="conv2")
    lrn2 = tf.nn.local_response_normalization(
            conv2,
            depth_radius=2,
            alpha=2e-05,
            beta=0.75,
            bias=1.0,
            name="norm2")
    pool2 = tf.nn.max_pool(
            lrn2,
            ksize=[1, 3, 3, 1],
            strides=[1, 2, 2, 1],
            padding="VALID",
            name="pool2")

    # 3rd layer
    conv3 = __conv__(pool2, 3, 3, 384, 1, 1, "conv3", padding="SAME", group=1)
    conv3 = tf.nn.relu(conv3, name="conv3")

    # 4th layer
    conv4 = __conv__(conv3, 3, 3, 384, 1, 1, "conv4", padding="SAME", group=2)
    conv4 = tf.nn.relu(conv4, name="conv4")

    # 5th layer
    conv5 = __conv__(conv4, 3, 3, 256, 1, 1, "conv5", padding="SAME", group=2)
    conv5 = tf.nn.relu(conv5, name="conv5")
    pool5 = tf.nn.max_pool(
            conv5,
            ksize=[1, 3, 3, 1],
            strides=[1, 2, 2, 1],
            padding="VALID",
            name="pool5")

    # flatten the output of the last layer so it can be used as input for a fully connected layer
    flat = tf.reshape(pool5, [-1, int(np.prod(pool5.get_shape()[1:]))])

    # 6th layer
    fc6 = __fc__(flat, 6 * 6 * 256, 4096, "fc6")
    drop6 = tf.nn.dropout(fc6, dropout)

    # 7th layer
    fc7 = __fc__(drop6, 4096, 4096, "fc7")
    drop7 = tf.nn.dropout(fc7, dropout)

    return __fc__(drop7, 4096, classes, "fc8", relu=False)


def load_weights(path, session, skip_layer):
    """
    Loads pretrained weights for the AlexNet model from file.
    The weights will be assigned within the current tensorflow session.

    :param path: filepath to file containing weights
    :param session: tensorflow session the model is loaded into
    :param skip_layer: list of layers by name to skip to be trained from scratch
    """
    # Load the weights into memory
    weights = np.load(path, encoding='bytes').item()
    # Loop over all layer names stored in the weights dict
    for layer in weights:
        # Check if layer should be trained from scratch
        if layer not in skip_layer:
            with tf.variable_scope(layer, reuse=True):
                # Assign weights/biases to their corresponding tf variable
                for data in weights[layer]:
                    w = tf.get_variable("weights", trainable=False)
                    session.run(w.assign(weights[layer][0]))
                    b = tf.get_variable('biases', trainable=False)
                    session.run(b.assign(weights[layer][1]))
