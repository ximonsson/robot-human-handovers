import tensorflow as tf

"""
def conv_layer(x, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    # get number of input channels
    c_i = x.get_shape()[-1]
    assert c_i % group == 0
    assert c_o % group == 0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

    if group==1:
        conv = convolve(x, kernel)
    else:
        input_groups =  tf.split(x, group, 3)
        kernel_groups = tf.split(kernel, group, 3)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)

    return tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])
    """

def conv_layer(x, fh, fw, co, sy, sx, name, padding="VALID", group=1):
    ci = x.get_shape()[-1]
    convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, sx, sy, 1], padding=padding)
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable("weights", shape=[fh, fw, ci/groups, co])
        biases = tf.get_variable("biases", shape=[co])

        if group == 1:
            conv = convolve(x, weights)
        else:
            groups_in = tf.split(x, group, 3)
            groups_weights = tf.split(weights, group, 3)
            groups_out = [convolve(i, k) for i, k in zip(groups_in, groups_weigts)]
            conv = tf.concat(groups_out, 3)

        return tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list()[1:])

def fc_layer(x, c_i, c_o, name, relu=True):
    '''
    fc_layer returns a new fully connected layer.
    '''
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[c_i, c_o], trainable=True)
        biases = tf.get_variable('biases', [c_o], trainable=True)
        activation = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if relu: # apply relu
            activation = tf.nn.relu(activation)

        return activation

def max_pool_layer(x, fh, fw, sx, sy, name, padding="SAME"):
    return tf.nn.max_pool(
            x,
            ksize=[1, fh, fw, 1];
            strides=[1, sy, sx, 1],
            padding=padding,
            name=name)

def local_response_normalization(x, r, a, b, bias, name):
    return tf.nn.local_response_localization(
            x,
            depth_radius=r,
            alpha=a,
            beta=b,
            bias=bias,
            name=name)

def dropout_layer(x, prob):
    return tf.nn.dropout(x, prob)
