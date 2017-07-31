#!/usr/bin/env python
# -*- coding: utf8 -*-

from __future__ import print_function

import os.path
import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt

dir_name = os.path.abspath(os.path.dirname(__file__))
libs_path = os.path.join(dir_name, 'libs')
sys.path.insert(0, libs_path)

from dataGenerator import DataGenerator
from batchGenerator import BatchGenerator

# Parameters
DATA_TIMESTEP = int(1e6)
BASE_PERIOD = 40

INPUT_SIZE = 1
INPUT_STEP = 20
OUTPUT_SIZE = 1
OUTPUT_STEP = 1

# Optimization Paramters
learning_rate = 0.001
training_iters = 1e6
batch_size = 128
display_step = 10

# Network Parameters
n_input = INPUT_SIZE
n_steps = INPUT_STEP
n_hidden = 128
n_output = OUTPUT_SIZE

# Import Data
train_data = DataGenerator(DATA_TIMESTEP, INPUT_SIZE, BASE_PERIOD)
train_data.toIOSet(INPUT_STEP, OUTPUT_STEP)
# data = MeteorolData()
# Plot wind speed data
# speed = data.windSpeed.tolist()


# Define input/output place
x = tf.placeholder('float', [None, n_input, n_steps])
y = tf.placeholder('float', [None, n_output])

# Define weights (fully connection layer to output)
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_output]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_output]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 2)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

predit = RNN(x, weights, biases)

# Define loss and optimizer
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
cost = tf.losses.mean_squared_error(predit, y)

# Use default learning rate
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# Use exponential decay learning rate
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(10.0, global_step, 5000, 0.1)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
gradients, v = zip(*optimizer.compute_gradients(cost))
gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
optimizer = optimizer.apply_gradients(zip(gradients, v), global_step = global_step)

# Evaluate model
# correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
accuracy = tf.reduce_mean(tf.square(y-tf.cast(predit, tf.float32)))

# Initializing the variables
init = tf.global_variables_initializer()

# Prepare Batches
train_batch = BatchGenerator(train_data.inputSet, train_data.outputSet, batch_size)

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    acc_log = []
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = train_batch.nextBatch()
        # Reshape data to get 28 seq of 28 elements
        # batch_x = np.asarray(batch_x)
        # batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        batch_y = batch_y.reshape([batch_size, n_output])
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            acc_log.append(acc.item())
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # # Calculate accuracy for 128 mnist test images
    # test_len = 128
    # test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    # test_label = mnist.test.labels[:test_len]
    # print("Testing Accuracy:", \
    #     sess.run(accuracy, feed_dict={x: test_data, y: test_label}))

    test_data = DataGenerator(200, INPUT_SIZE, BASE_PERIOD)
    test_data.toIOSet(INPUT_STEP, OUTPUT_STEP)
    test_batch = BatchGenerator(test_data.inputSet, test_data.outputSet, batch_size)

    batch_x, batch_y = test_batch.nextBatch()
    batch_y = batch_y.reshape((batch_size, n_input))
    y_hat = sess.run(predit, feed_dict={x: batch_x, y: batch_y})

plt.subplot(2,1,1)
plt.plot(range(200), test_data.data.flatten())
plt.plot(range(20,len(y_hat)+INPUT_STEP),y_hat)
plt.subplot(2,1,2)
plt.plot(acc_log)
plt.show()
