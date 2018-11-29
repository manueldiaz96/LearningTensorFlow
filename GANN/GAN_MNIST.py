#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import random
import numpy as np
from matplotlib import pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

def conv2d(x, W):
	return tf.nn.conv2d(input=x, filter=W, strides=[1,1,1,1], padding='SAME')

def avg_pool_2x2(x):
	return tf.nn.avg_pool(x, ksize=[1,2,2,1], padding='SAME')

def discriminator(x_image, reuse=False):
	with tf.variable_scope('discriminator') as scope:
		if reuse:
			tf.get_variable_scope().reuse_variables()

		#First Conv and Pool Layers
		W_conv1 = tf.get_variable('d_wconv1', [5, 5, 1, 8], initializer=tf.truncated_normal_initializer(stddev=0.02))
		b_conv1 = tf.get_variable('d_bconv1', [8], initializer=tf.constant_initializer(0))
		h_conv1 = tf.nn.leaky_relu(conv2d(x_image, W_conv1)+b_conv1)
		h_pool1 = avg_pool_2x2(h_conv1)

		#Second Conv and Pool Layers
		W_conv2 = tf.get_variable('d_wconv2', [5, 5, 8, 16], initializer=tf.truncated_normal_initializer(stddev=0.02))
		b_conv2 = tf.get_variable('d_bconv1', [16], initializer=tf.constant_initializer(0))
		h_conv2 = tf.nn.leaky_relu(conv2d(h_pool1, W_conv2)+b_conv2)
		h_pool2 = avg_pool_2x2(h_conv2)

		#First Fully Connected Layer
		W_fc1 = tf.get_variable('d_wfc1', [7 * 7 * 16, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
		b_fc1 = tf.get_variable('d_bfc1', [32], initializer=tf.constant_initializer(0))
		h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*16])
		h_fc1 = tf.nn.leaky_relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

		#Second fully Connected Layer
		W_fc2 = tf.get_variable('d_wfc2', [32,1], initializer=tf.truncated_normal_initializer(stddev=0.02))
		b_fc2 = tf.get_variable('d_bfc2', [1], initializer=tf.constant_initializer(0))

		#Final Layer
		y_conv = (tf.matmul(h_fc1, W_fc2) + b_fc2)

	return y_conv


mnist = input_data.read_data_sets("MNIST_data/")
x_train = mnist.train.images[:55000,:]
print(x_train.shape)

randomNum = random.randint(0,55000)
image = x_train[randomNum].reshape([28,28])
plt.imshow(image, cmap=plt.get_cmap("gray_r"))
plt.show()