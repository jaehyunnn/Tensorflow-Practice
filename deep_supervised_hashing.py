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

def fully_connected(value, ouput_shape name='fully_connected', with_w=False):
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
        norm1 = tf.nn.lrn(value, depth_radius=depth_radius, bias=1.0, alpha=alpha, beta=beta)
        return norm1


def discriminator(image, hashing_bits, reuse=False, name='discriminator'):
    with tf.name_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        conv1 = conv2d(image, output_dim=32, name='d_conv1')
        relu1 = relu(pool(conv1, name='d_lrn1'), name='d_relu1')
        conv2 = conv2d(lrn(relu2, name='d_lrn1'), output_dim=32, name='d_conv2')
        relu2 = relu(pool_avg(conv2, name='d_pool2'), name='d_relu2')
        conv3 = conv2d(lrn(relu2, name='d_lrn2'), output_dim=64, name='d_conv3')
        pool3 = pool_avg(relu(conv3, name='d_relu3'), name='d_pool3')
        relu_ip1 = relu(fully_connected(pool3, ouput_shape=500, name='d_ip1'), name='d_relu4')
        ip2 = fully_connected(relu_ip1, output_shape=hashing_bits, name='d_ip2')

        return ip2

def read_cifar10_data():
    data_dir = CURRENT_DIR + '/data/cifar-10-batches-py/'
    train_name = 'data_batch_'
    test_name = 'test_batch'
    train_X = None
    train_Y = None
    test_X = None
    test_Y = None

    for i in range(1,6):
        file_path = data_dir + train_name + str(i)
        with open(file_path, 'rb') as fo:
            dict = cPickle.load(fo)
            if train_X is None:
                train_X = dict['data']
                train_Y = dict['labels']
            else:
                train_X = np.concatenate((train_X, dict['data']), axis=0)
                train_Y = np.concatenate((train_Y, dict['labels']), axis=0)

    file_path = data_dir + test_name
    with open(file_path, 'rb') as fo:
        dict = cPickle.load(fo)
        test_X = dict['data']
        test_Y = dict['labels']
    train_X = train_X.reshape((50000, 3, 32, 32)).transpose(0, 2, 3, 1).astype(np.float))
    test_X = test_X.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1).astype(np.float)

    train_y_vec = np.zeros((len(train_Y), 10), dtype=np.float)
    test_y_vec = np.zeros((len(test_Y), 10), dtype=np.float)
    for i, label in enumerate(train_Y):
        train_y_vec[i, int(train_Y[i])] = 1.
    for i, label in enumerate(test_Y):
        test_y_vec[i, int(test_Y[i])] = 1.

    return train_X/255., train_y_vec, test_X/255., test_y_vec

def hashing_loss(image, label, alpha, m):
    D = discriminator(image, HASHING_BITS)
    w_label = tf.matmul(label, label, False, True)

    r = tf.reshape(tf.reduce_sum(D*D, 1), [-1, 1])
    p2_distance = r - 2*tf.matmul(D, D, False, True) + tf.transpose(r)
    temp = w_label*p2_distance + (1-w_label)*tf.maximum(m-p2_distance, 0)

    regularizer = tf.reduce_sum(tf.abs(tf.abs(D) - 1))
    d_loss = tf.reduce_sum(temp)/(BATCH_SIZE*(BATCH_SIZE-1)) + alpha*regularizer/BATCH_SIZE
    return d_loss