# -*- coding: utf-8 -*-

"""
In this file we try out InteractiveSession instead of Session,
which allows us to run the model interactively.
We build a convolutional neural network to classify MNIST dataset.

Placeholder:
    - x: input images
    - y: predicted labels
    - y_: real labels
Variable:
    - W_conv1, b_conv1: convolution layer parameters
    - W_fc[k], b_fc1[k]: fully-connected layer parameters
"""
# load data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

sess = tf.InteractiveSession()

# Create handlers for parameters.
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    # Since we use ReLu as our activation function,
    # a slightly positive bias can, to some extent, reduce dead neurons.
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Convolution
# Filter
# - stride: 1
# - padding: 0

def conv2d(x, W):
    """
    :param x: input
    :param W: filter parameters
    :param strides: len(4), strides along every axis of input
    """
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

# Pooling
# max pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# Build model
# - input: (batch_size, img_w, img_h, img_chn)
# - weight(filter): (patch_size, patch_size, input_channel, output_channel(filter_num))
#   (5,5,1,32)

# define weight placeholder
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

# reshape the input batch
x_image = tf.reshape(x, [-1,28,28,1])   #? -1

# Construct first layer
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Fully-Connected layer
# affine layer with 1024 neurons
W_fc1 = weight_variable([14*14*32, 1024])
b_fc1 = bias_variable([1024])

h_pool1_flat = tf.reshape(h_pool1, [-1, 14*14*32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1)+b_fc1)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y = tf.matmul(h_fc1, W_fc2) + b_fc2

# Dropout: to apply dropout between fc1 & fc2
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Compute loss
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))

# Construct training procedure
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# tf.argmax(): Returns the index with the largest value across axiss of a tensor.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize variables
sess.run(tf.global_variables_initializer())

# Train
for i in range(10000):   # 10000 iterations
    batch = mnist.train.next_batch(100)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1]})
        print("step %d, training accuracy %g" % (i, train_accuracy))

    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

print("test accuracy %g" % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels}))

"""
Running 10000 iterations with one-layer CNN results in pool accuracy.
"""
