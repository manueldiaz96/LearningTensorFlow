# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 09:46:23 2018

@author: Manuel Alejandro Diaz Zapata

https://blog.paperspace.com/implementing-gans-in-tensorflow/
https://github.com/aadilh/blogs/tree/new/basic-gans/basic-gans/code
"""
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

def get_y(x):
    return (x**2)+(4*x)

def sample_data(n=500, scale=100):
    data = []
    x = scale*(np.random.random_sample((n,))-0.5)
    
    for i in range(n):
        yi = get_y(x[i])
        data.append([x[i],yi])
        
    return np.array(data)

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def generator(Z, hsize=[16,16], reuse = False):
    
    '''
    This function takes in the placeholder for random samples (Z), 
    an array hsize for the number of units in the 2 hidden layers 
    and a reuse variable which is used for reusing the same layers. 
    Using these inputs it creates a fully connected neural network 
    of 2 hidden layers with given number of nodes. The output of 
    this function is a 2-dimensional vector which corresponds to 
    the dimensions of the real dataset that we are trying to learn. 
    The above function can be easily modified to include more hidden 
    layers, different types of layers, different activation and 
    different output mappings.
    
    '''
    
    with tf.variable_scope("GAN/Generator", reuse=reuse):
        h1 = tf.layers.dense(Z, hsize[0], activation = tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1, hsize[1], activation = tf.nn.leaky_relu)
        out = tf.layers.dense(h2, 2)
        
    return out


def discriminator(X, hsize=[16,16], reuse=False):
    
    '''
    
    This function takes input placeholder for the samples from the 
    vector space of real dataset. The samples can be both real samples 
    and samples generated from the Generator network. Similar to the 
    Generator network above it also takes input hsize and reuse. We 
    use 3 hidden layers for the Discriminator out of which first 2 
    layers size we take input. We fix the size of the third hidden 
    layer to 2 so that we can visualize the transformed feature space 
    in a 2D plane as explained in the later section. The output of this 
    function is a logit prediction for the given X and the output of 
    the last layer which is the feature transformation learned by 
    Discriminator for X. The logit function is the inverse of the sigmoid 
    function which is used to represent the logarithm of the odds (ratio 
    of the probability of variable being 1 to that of it being 0).
    
    '''
    
    with tf.variable_scope("GAN/Discriminator", reuse=reuse):
        h1 = tf.layers.dense(X, hsize[0], activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1, hsize[1], activation=tf.nn.leaky_relu)
        h3 = tf.layers.dense(h2, 2)
        out = tf.layers.dense(h3, 1)
        
    return out, h3


print "ver" ,tf.__version__

data = sample_data()



# For the purpose of training we define the following placeholders 
# X and Z for real samples and random noise samples respectively:
X = tf.placeholder(tf.float32, [None, 2])
Z = tf.placeholder(tf.float32, [None, 2])

#create the graph for generating samples from Generator network and feeding real 
#and generated samples to the Discriminator network
G_sample = generator(Z)
r_logits, r_rep = discriminator(X)
f_logits, g_rep = discriminator(G_sample, reuse=True)

#Loss functions
'''
These losses are sigmoid cross entropy based losses using the equations we 
defined above. This is a commonly used loss function for so-called discrete 
classification. It takes as input the logit (which is given by our discriminator 
network) and true labels for each sample. It then calculates the error for 
each sample. We are using the optimized version of this as implemented by 
TensorFlow which is more stable then directly taking calculating cross entropy.

https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

'''
disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = r_logits, labels = tf.ones_like(r_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.zeros_like(f_logits)))
gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = f_logits, labels = tf.ones_like(f_logits)))

#Optimizers
'''
optimizers for the two networks using the loss functions defined above and 
scope of the layers defined in the generator and discriminator functions. We 
use RMSProp Optimizer for both the networks with the learning rate as 0.001. 
Using the scope we fetch the weights/variables for the given network only

'''

gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "GAN/Generator")
disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "GAN/Discriminator")

gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(gen_loss, var_list=gen_vars)
disc_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(disc_loss, var_list=disc_vars)

batch_size = 500

sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

dloss_plot = []
gloss_plot = []

import time
start_time = time.time()

iterations = 20000
fig_generator, gen_plot = plt.subplots()

xax = gen_plot.scatter(data[:,0], data[:,1])
diff_gen_datos = 0


for i in range(iterations+1):
    X_batch = sample_data(n = batch_size)
    Z_batch = sample_Z(batch_size, 2)
    _, dloss = sess.run([disc_step, disc_loss], feed_dict={X:X_batch, Z: Z_batch})
    _, gloss = sess.run([gen_step, gen_loss], feed_dict={Z: Z_batch})

    g_plot = sess.run(G_sample, feed_dict={Z: Z_batch})
    diff_gen_datos = np.mean((data-g_plot))
    print(r_logits)
    
    print ("Iterations: %d\t Discriminator loss: %.4f\t Generator loss: %.4f\t Diff: %.3f"%(i,dloss,gloss,diff_gen_datos))
    dloss_plot.append(dloss)
    gloss_plot.append(gloss)



g_plot = sess.run(G_sample, feed_dict={Z: Z_batch})
xax = gen_plot.scatter(data[:,0], data[:,1], color = "blue", label="Real Data")
gax = gen_plot.scatter(g_plot[:,0],g_plot[:,1], color = "green", label="Generated Data")
gen_plot.legend()
plt.plot()

print("%s seconds, %s iterations" % (time.time() - start_time, iterations))

#min_max_losses = (np.min((np.mean(dloss_plot), np.mean(gloss_plot))), np.max((np.mean(dloss_plot), np.mean(gloss_plot))))


fig_losses, losses_plot = plt.subplots()
dplt = losses_plot.plot(dloss_plot, color = "green", label='Discriminator Loss')
gplt = losses_plot.plot(gloss_plot, color = "blue", label='Generator Loss')
losses_plot.legend()
losses_plot.set_title("Losses")


plt.show()
