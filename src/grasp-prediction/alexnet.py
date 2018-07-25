'''
Implementation of the AlexNet model taken from:
    https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html
    http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
'''

import numpy as np
from model import conv_layer, fc_layer, max_pool_layer, local_response_normalization, dropout_layer

def load_weights(path, skip_layer):
    # Load the weights into memory
    weights_dict = np.load(path, encoding='bytes').item()
    # Loop over all layer names stored in the weights dict
    for op_name in weights_dict:
        # Check if layer should be trained from scratch
        if op_name not in skip_layer:
            with tf.variable_scope(op_name, reuse=True):
                # Assign weights/biases to their corresponding tf variable
                for data in weights_dict[op_name]:
                    # Biases
                    if len(data.shape) == 1:
                        var = tf.get_variable('biases', trainable=False)
                        session.run(var.assign(data))
                    # Weights
                    else:
                        var = tf.get_variable('weights', trainable=False)
                        session.run(var.assign(data))

# TODO solve nclasses (number of classes to output)
def model(path, nclasses):
    x = tf.placeholder(tf.float32, (None,) + np.zeros((1, 227, 227, 3)).astype(float32))

    # 1st layer
    conv1 = conv_layer(x, 11, 11, 96, 4, 4, "conv1" padding="SAME", group=1)
    conv1 = tf.relu(conv1)
    lrn1 = local_response_normalization(conv1, 2, 2e-05, 0.75, 1.0, "norm1")
    pool1 = max_pool_layer(lrn1, 3, 3, 2, 2, "pool1", padding="VALID")

    # 2nd layer
    conv2 = conv_layer(pool1, 5, 5, 256, 1, 1, "conv2", padding="SAME", group=2)
    conv2 = tf.relu(conv2)
    lrn2 = local_response_normalization(conv2, 2, 2e-05, 0.75, 1.0, "norm2")
    pool2 = max_pool_layer(lrn2, 3, 3, 2, 2, "pool2", padding="VALID")

    # 3rd layer
    conv3 = conv_layer(pool2, 3, 3, 384, 1, 1, "conv3", padding="SAME", group=1)
    conv3 = tf.relu(conv3)

    # 4th layer
    conv4 = conv_layer(conv3, 3, 3, 384, 1, 1, "conv4", padding="SAME", group=2)
    conv4 = tf.relu(conv4)

    # 5th layer
    conv5 = conv_layer(conv4, 3, 3, 256, 1, 1, "conv5", padding="SAME", group=2)
    conv5 = tf.relu(conv5)
    pool5 = max_pool_layer(conv5, 3, 3, 2, 2, "pool5", padding="VALID")

    # 6th layer
    tmp = tf.reshape(pool5, [-1, int(prod(pool5.get_shape()[1:]))])
    fc6 = fc_layer(6 * 6 * 256, 4096, "fc6")
    #drop6 = dropout_layer(fc6, 0.5)

    # TODO check if there is supposed to be dropout here or not

    # 7th layer
    fc7 = fc_layer(fc6, 4096, 4096, "fc7")
    #drop7 = dropout_layer(fc7, 0.5)

    # TODO check if there is supposed to be dropout here or not

    return fc_layer(fc7, 4096, nclasses, "fc8", relu=False)
