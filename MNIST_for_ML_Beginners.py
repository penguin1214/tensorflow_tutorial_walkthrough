# -*- coding: utf-8 -*-

# load data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

# x is not a specific value, but a placeholder to place input data when computing
x = tf.placeholder(tf.float32, [None, 784])

# weights and biases are parameters varies with time, so be define them as 'variables'
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# model
y = tf.nn.softmax(tf.matmul(x, W) + b)

# loss function
# y_ is correct labels, None means
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))

# train
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# init variables
init = tf.global_variables_initializer()

# define session
sess = tf.Session()

# run the initialization
sess.run(init)

# run the training procedure
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# evaluation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# placeholder is not actual data, we use feed_dict parameter to 'assign' data to placeholder used in ops.
