import gym
import numpy as np
import random
import tensorflow as tf
from parameters import HP

#TODO read momentum and feed parameters


class CNN:
	def __init__(self, env):
        self.sess = tf.Session()
        self.input_width = 84
        self.input_stack_size = 4
        self.definePlaceHolders()
		self.defineNNArchitecture()
		self.defineLossAndTrainer()
        self.sess = tf.Session()
		init = tf.global_variables_initializer()
		self.sess.run(init)
		self.saver = tf.train.Saver()


    def preprocess(frame):
        data = imresize(data, (WIDTH, WIDTH))
        data = np.dot(data, [0.2126, 0.7152, 0.0722])
        return data

    #using vectorize call preprocess on all elements then convert in to tuple and dstack them
    def stackFrames(inputs):
        # return np.dstack((self.preprocess for inputVal in inputs))
        pass

	def definePlaceHolders(self):
		self.X = tf.placeholder(
            shape=[None, self.input_width*self.input_width*self.input_stack_size],
            dtype=tf.float32)
    	self.Q = tf.placeholder(shape=[None],dtype=tf.float32)
    	    self.targetActionMask = tf.placeholder(shape=[None, self.output_size],dtype=tf.float32)

    def defineLossAndTrainer(self):
		temp = tf.reduce_sum(tf.multiply(self.Qprime, self.targetActionMask), 1)
		print 'temp shape', temp.shape
		print 'Q shape ',self.Q.shape
		self.loss = tf.reduce_mean(tf.square(temp -  self.Q))
		for weightRegul in self.weights[0::2]:
			self.loss += HP['regularization_factor'] * tf.reduce_sum(tf.square(weightRegul))
		trainer = tf.train.RMSPropOptimizer(
            learning_rate = HP['learning_rate'],
            epsilon=0.01,
            decay=0.95,
            momentum=0.95)
		self.step = trainer.minimize(self.loss)

	def defineCNNArchitecture(self, input):
		conv1 = tf.layers.conv2d(
      		inputs=input_layer,
      		filters=32,
            strides=[1, 4, 4, 1],
      		kernel_size=[8, 8],
      		activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(
      		inputs= conv1,
      		filters=64,
            strides=[1, 2, 2, 1],
      		kernel_size=[4, 4],
      		activation=tf.nn.relu)
    	conv3 = tf.layers.conv2d(
      		inputs=conv2,
      		filters=64,
            strides=[1, 1, 1, 1],
      		kernel_size=[3, 3],
      		activation=tf.nn.relu)
        flatten = tf.reshape(conv3, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=flatten, units=512, activation=tf.nn.relu)
        output = tf.layers.dense(inputs=dense, units=env.output_size)
