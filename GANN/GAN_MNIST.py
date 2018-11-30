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

def generator(z, batch_size, z_dim, reuse=False):
	with tf.variable_scope('generator') as scope:
		if reuse:
			tf.get_variable_scope().reuse_variables()

		g_dim = 64 #Number of filters of first layer of generator
		c_dim = 1 #Color dimension of output (grayscale)
		s = 28 #Output size of image
		s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16) #These values will help sine we want to slowly upscale the image

		h0 = tf.reshape(z, [batch_size, s16+1, s16+1, 25])
		h0 = tf.nn.leaky_relu(h0)
		#Dims of h0: batch_size x 2 x 2 x 25


		#First Deconv layer
		output1_shape = [batch_size, s8, s8, g_dim*4]
		W_conv1 = tf.get_variable('g_wconv1', [5, 5, output1_shape[-1], int(h0.get_shape()[-1])], initializer=tf.truncated_normal_initializer(stddev=0.1))
		b_conv1 = tf.get_variable('g_bconv1', [output1_shape[-1]], initializer=tf.constant_initializer(.1))
		H_conv1 = tf.nn.conv2d_transpose(h0, W_conv1, output_shape=output1_shape, strides=[1, 2, 2, 1], padding='SAME') + b_conv1
		H_conv1 = tf.contrib.layers.batch_norm(inputs = H_conv1, center = True, scale = True, is_training=True, scope='g_bn1')
		H_conv1 = tf.nn.leaky_relu(H_conv1)
		#Dims of H_conv1: batch_size x 3 x 3 x 256


		#Second Deconv layer
		output2_shape = [batch_size, s4 - 1, s4 -1, g_dim*2]
		W_conv2 = tf.get_variable('g_wconv2', [5, 5, output2_shape[-1], int(H_conv1.get_shape()[-1])], initializer=tf.truncated_normal_initializer(stddev=0.1))
		b_conv2 = tf.get_variable('g_bconv2', [output2_shape[-1]], initializer=tf.constant_initializer(.1))
		H_conv2 = tf.nn.conv2d_transpose(H_conv1, W_conv2, output_shape=output2_shape, strides=[1, 2, 2, 1], padding='SAME') + b_conv2
		H_conv2 = tf.contrib.layers.batch_norm(inputs = H_conv2, center = True, scale = True, is_training=True, scope='g_bn2')
		H_conv2 = tf.nn.leaky_relu(H_conv2)
		#Dims of H_conv2: batch_size x 6 x 6 x 128


		#Third Deconv layer
		output3_shape = [batch_size, s2 - 2, s2 - 2, g_dim*1]
		W_conv3 = tf.get_variable('g_wconv3', [5, 5, output3_shape[-1], int(H_conv2.get_shape()[-1])], initializer=tf.truncated_normal_initializer(stddev=0.1))
		b_conv3 = tf.get_variable('g_bconv3', [output3_shape[-1]], initializer=tf.constant_initializer(.1))
		H_conv3 = tf.nn.conv2d_transpose(H_conv2, W_conv3, output_shape=output3_shape, strides=[1, 2, 2, 1], padding='SAME') + b_conv3
		H_conv3 = tf.contrib.layers.batch_norm(inputs = H_conv3, center = True, scale = True, is_training=True, scope='g_bn3')
		H_conv3 = tf.nn.leaky_relu(H_conv3)
		#Dims of H_conv3: batch_size x 12 x 12 x 64


		#Fourth Deconv layer
		output4_shape = [batch_size, s, s, c_dim]
		W_conv4 = tf.get_variable('g_wconv4', [5, 5, output4_shape[-1], int(H_conv3.get_shape()[-1])], initializer=tf.truncated_normal_initializer(stddev=0.1))
		b_conv4 = tf.get_variable('g_bconv4', [output4_shape[-1]], initializer=tf.constant_initializer(.1))
		H_conv4 = tf.nn.conv2d_transpose(H_conv3, W_conv4, output_shape=output4_shape, strides=[1, 2, 2, 1], padding='VALID') + b_conv4
		H_conv4 = tf.nn.tanh(H_conv4)
		#Dims of H_conv4: batch_size x 28 x 28 x 1
		
	return H_conv4




mnist = input_data.read_data_sets("MNIST_data/")
x_train = mnist.train.images[:55000,:]
print(x_train.shape)

randomNum = random.randint(0,55000)
image = x_train[randomNum].reshape([28,28])
plt.imshow(image, cmap=plt.get_cmap("gray_r"))
plt.show()