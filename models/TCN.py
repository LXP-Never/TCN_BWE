# -*- coding: utf-8 -*-
import tensorflow as tf


def causal_padding(x, padding=(1, 1)):
    """Pads the middle dimension of a 3D tensor.
    # Arguments
        x: Tensor or variable.
        padding: Tuple of 2 integers, how many zeros to
            add at the start and end of dim 1.
    # Returns
        A padded 3D tensor.
    """
    assert len(padding) == 2
    pattern = [[0, 0], [padding[0], padding[1]], [0, 0]]
    return tf.pad(x, pattern, mode="REFLECT")



def TCN_block(input, filters, kernel_size, dilation_rate, dropout_rate=0.0, activation=tf.nn.leaky_relu,
              regularizer_scale=0.01, training=True):
    with tf.variable_scope("TCN_block_conv0"):
        padding = (kernel_size - 1) * dilation_rate
        conv0 = causal_padding(input, padding=[padding, 0])
        conv0 = tf.layers.conv1d(conv0, filters=filters, kernel_size=kernel_size, strides=1, activation=None,
                                 padding="valid", dilation_rate=dilation_rate,
                                 kernel_initializer=tf.orthogonal_initializer())
        # kernel_regularizer = tf.contrib.layers.l1_regularizer(regularizer_scale)
        # conv -> activation -> bn -> dropout(可能更加有效)
        # conv -> bn -> activation -> dropout
        conv0 = tf.layers.batch_normalization(conv0, training=training)
        conv0 = activation(conv0)
        # conv0 = tf.layers.dropout(conv0, rate=dropout_rate)
        # print('conv0: ', conv0.get_shape())  # (?, 4096, 64)
    with tf.variable_scope("TCN_block_conv1"):
        conv1 = tf.layers.conv1d(conv0, filters=filters * 2, kernel_size=kernel_size, strides=1, activation=None,
                                 padding="same", kernel_initializer=tf.orthogonal_initializer())
        conv1 = tf.layers.batch_normalization(conv1, training=training)
        # conv1 = tf.layers.dropout(conv1, rate=dropout_rate)
        # print('D-Block: ', conv1.get_shape())  # (?, 2048, 128)
    if input.get_shape().as_list() == conv1.get_shape().as_list():
        shortcut = input
    else:
        with tf.variable_scope("shortcut_conv"):
            shortcut = tf.layers.conv1d(input, filters=filters * 2, kernel_size=1, activation=None,
                                        padding="same", kernel_initializer=tf.orthogonal_initializer())
            # shortcut = tf.nn.leaky_relu(shortcut, alpha=0.2)
            # print('shortcut: ', shortcut.get_shape())  # (?, 2048, 128)
    return activation(tf.add(shortcut, conv1))


def SubPixel1D(I, r):
    """One-dimensional subpixel upsampling layer

    Calls a tensorflow function that directly implements this functionality.
    We assume input has dim (batch, width, r)
    """
    with tf.name_scope('subpixel'):
        X = tf.transpose(I, [2, 1, 0])  # (r, w, b)
        X = tf.batch_to_space_nd(X, [r], [[0, 0]])  # (1, r*w, b)
        X = tf.transpose(X, [2, 1, 0])
        return X


def TCN_model(inputs, dropout_rate, is_training):
    with tf.variable_scope("first"):
        x = tf.layers.conv1d(inputs, filters=64, kernel_size=9, activation=None, padding="same",
                             kernel_initializer=tf.orthogonal_initializer(), strides=2)
        x = tf.nn.leaky_relu(x, alpha=0.2)
    with tf.variable_scope("TCN_conv0"):
        x = TCN_block(x, filters=64, kernel_size=9, dilation_rate=2 ** 0, dropout_rate=dropout_rate,
                      training=is_training)
    with tf.variable_scope("TCN_conv1"):
        x = TCN_block(x, filters=64, kernel_size=9, dilation_rate=2 ** 1, dropout_rate=dropout_rate,
                      training=is_training)
    # ---------------------------------------------------------------------------
    with tf.variable_scope("TCN_conv2"):
        x = TCN_block(x, filters=64, kernel_size=9, dilation_rate=2 ** 2, dropout_rate=dropout_rate,
                      training=is_training)
    with tf.variable_scope("TCN_conv3"):
        x = TCN_block(x, filters=64, kernel_size=9, dilation_rate=2 ** 3, dropout_rate=dropout_rate,
                      training=is_training)
    with tf.variable_scope("TCN_conv4"):
        x = TCN_block(x, filters=64, kernel_size=9, dilation_rate=2 ** 4, dropout_rate=dropout_rate,
                      training=is_training)
    with tf.variable_scope('lastconv'):
        x_last = tf.layers.conv1d(x, filters=2, kernel_size=9, activation=None, padding="same",
                                  kernel_initializer=tf.random_normal_initializer(mean=0, stddev=1e-3))
        x_last = SubPixel1D(x_last, r=2)
        print("last_layer", x_last.get_shape())
    outputs = tf.add(x_last, inputs)
    return outputs


def dilation_conv_model(inputs, dropout_rate=0.0, is_training=True):
    with tf.variable_scope("first"):
        x = tf.layers.conv1d(inputs, filters=64, kernel_size=9, activation=None, padding="same",
                             kernel_initializer=tf.orthogonal_initializer(), strides=2)
        x = tf.nn.leaky_relu(x, alpha=0.2)
    with tf.variable_scope("conv1"):
        x = tf.layers.conv1d(x, filters=64, kernel_size=9, strides=1, padding="same",
                             dilation_rate=2, kernel_initializer=tf.orthogonal_initializer())
        x = tf.nn.leaky_relu(x, alpha=0.2)
        print("conv1", x.shape)
    with tf.variable_scope("conv2"):
        x = tf.layers.conv1d(x, filters=64, kernel_size=9, strides=1, padding="same",
                             dilation_rate=3, kernel_initializer=tf.orthogonal_initializer())
        x = tf.nn.leaky_relu(x, alpha=0.2)
        print("conv2", x.shape)
    with tf.variable_scope("conv3"):
        x = tf.layers.conv1d(x, filters=64, kernel_size=9, strides=1, padding="same",
                             dilation_rate=4, kernel_initializer=tf.orthogonal_initializer())
        x = tf.nn.leaky_relu(x, alpha=0.2)
        print("conv3", x.shape)
    with tf.variable_scope("conv4"):
        x = tf.layers.conv1d(x, filters=64, kernel_size=9, strides=1, padding="same",
                             dilation_rate=5, kernel_initializer=tf.orthogonal_initializer())
        x = tf.nn.leaky_relu(x, alpha=0.2)
    with tf.variable_scope("conv5"):
        x = tf.layers.conv1d(x, filters=64, kernel_size=9, strides=1, padding="same",
                             dilation_rate=6, kernel_initializer=tf.orthogonal_initializer())
        x = tf.nn.leaky_relu(x, alpha=0.2)
    with tf.variable_scope('lastconv'):
        x_last = tf.layers.conv1d(x, filters=2, kernel_size=9, activation=None, padding="same",
                                  kernel_initializer=tf.random_normal_initializer(mean=0, stddev=1e-3))
        x_last = SubPixel1D(x_last, r=2)
        print("last_layer", x_last.get_shape())
    outputs = tf.add(x_last, inputs)
    return outputs





# test:
x = tf.get_variable(name="aaa", shape=[32, 8192, 1], dtype=tf.float32,
                    initializer=tf.random_normal_initializer(mean=0, stddev=1e-3))  # batch, seq_len, dim
x = TCN_model(x,dropout_rate=0.0,is_training=True)

print(x.shape)  # (32, 2048, 1)



