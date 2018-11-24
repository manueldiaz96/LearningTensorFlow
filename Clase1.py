#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf

'''
#Constantes
a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0)
total = a + b


print a
print b
print total

print sess.run(total)
print sess.run(a)
print sess.run(b)

'''
'''
#Variables
x = tf.constant(35, name='x')
y = tf.Variable(x+5, name='y')

init = tf.global_variables_initializer()
sess = tf.Session()

#inicializador de variables
sess.run(init)

print "x", sess.run(x)
print "y", sess.run(y)

'''
#Placeholders

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y

sess = tf.Session()
print sess.run(z, feed_dict={x:3, y:4.5})
print sess.run(z, feed_dict={x: [1,3], y:[2,4]})