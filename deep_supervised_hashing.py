import os
import numpy as np
import tensorflow as tf
import cPickle

batch_size = 200
learining_rate = 0.001
epoch = 600
load_model = False
trainable = True
hashing_bits = 12
current_dir = os.getcwd()

def bias(name, shape, bias_start=0.0, trainable=True):
    dtype = tf.float32
    var = tf.get_variable(name, shape, tf.float32, trainable=trainable, initializer=tf.constant_initializer(bias_start, dtype=dtype))
    return var

def weight(name, shape, stddev=0.02, trainable=True):
    dtype = tf.float32
    var = tf.get_variable(name, shape, tf.float32, trainable=trainable, initializer=tf.random_normal_initializer(stddev=stddev, dtype=dtype))

    return var

def fully_connected(value, ouput_shapen name='fully_connected', with_w=False):
    value = tf.reshape(value, [BATCH_SIZE, -1])
    shape = value.get_shape().as_list()

    with tf.variable_scope(name):
        weights = weight('weights', [shape[1], ouput_shape], 0.02)
        biases = bias('biases', [output_shape], 0.0)

    if with_w:
        return tf.matmul(value, weights) + biases, weights, biases
    else:
        return tf.matmul(value, weights) + biases

def relu(value, name='relu'):
    with tf.variable_scope(name):
        return tf.nn.relu(value)

def conv2d(value, output_dim, k_h=5, k_w=5, strides=[1, 1, 1, 1], name='conv2d'):
    with tf.variable_scope(name):
        weights = weight('weights', [k_h, k_w, value.get_shape()[-1], output_dim])
        conv = tf.nn.conv2d(value, weights, strides=strides, padding='SAME')
        biases = bias('biases', [output_dim])
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

def pool(value, k_size=[1, 3, 3, 1], strides=[1, 2, 2, 1], name='pool1'):
    with tf.variable_scope(name):
        pool = tf.nn.max_pool(value, ksize=k_size, strides=strides, padding='VALID')
        return pool

def lrn(value, depth_radius=1, alpha=5e-05, beta=0.75, name='lrn1'):
    with tf.variable_scope(name):
        norm1 = tf.nn.lrn()