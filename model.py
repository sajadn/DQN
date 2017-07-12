import gym
import numpy as np
import random
import tensorflow as tf
from parameters import HP


# a neural network with two hidden layers (128 nodes for each)
class NN:
	def __init__(self, env):
		self.sess = tf.Session()
		self.input_size = env.observation_space.shape[0]
    		self.output_size = env.action_space.n
		self.definePlaceHolders()
		self.defineNNArchitecture()
		self.defineLossAndTrainer()
		init = tf.global_variables_initializer()
		self.sess.run(init)
		self.saver = tf.train.Saver()


	def definePlaceHolders(self):
		self.X = tf.placeholder(shape=[None, self.input_size],dtype=tf.float32)
		self.Q = tf.placeholder(shape=[None],dtype=tf.float32)
	    	self.targetActionMask = tf.placeholder(shape=[None, self.output_size],dtype=tf.float32)

	def defineNNArchitecture(self):
		W1 = tf.Variable(tf.random_uniform([self.input_size, 128],0,0.01))
		b1 = tf.Variable(tf.random_uniform([1, 128],0,0.01))
		A1 = tf.add(tf.matmul(self.X, W1), b1)
		Z1 = tf.nn.relu(A1)
		W2 = tf.Variable(tf.random_uniform([128, 128],0,0.01))
		b2 = tf.Variable(tf.random_uniform([1, 128],0,0.01))
		A2 = tf.add(tf.matmul(Z1, W2), b2)
		Z2 = tf.nn.relu(A2)
		W3 = tf.Variable(tf.random_uniform([128, self.output_size],0,0.01))
		b3 = tf.Variable(tf.random_uniform([1, self.output_size],0,0.01))
		self.Qprime = tf.add(tf.matmul(Z2, W3), b3)
		self.P = tf.argmax(self.Qprime, 1)
		self.Qmean = tf.reduce_mean(self.Qprime)
		self.weights = [W1, b1, W2, b2, W3, b3]

	def defineLossAndTrainer(self):
		temp = tf.reduce_sum(tf.multiply(self.Qprime, self.targetActionMask), 1)
		print 'temp shape', temp.shape
		print 'Q shape ',self.Q.shape
		self.loss = tf.reduce_mean(tf.square(temp -  self.Q))
		for weightRegul in self.weights[0::2]:
			self.loss += HP['regularization_factor'] * tf.reduce_sum(tf.square(weightRegul))
		trainer = tf.train.GradientDescentOptimizer(learning_rate = HP['learning_rate'])
		self.step = trainer.minimize(self.loss)


	def predictAction(self, input):
		return self.sess.run(self.P, feed_dict= { self.X : input })

	def executeStep(self, input_val, output, target_action_mask):
		_, loss, Qmean_val = self.sess.run([self.step, self.loss, self.Qmean],
			feed_dict = {
				self.X: input_val,
			 	self.Q: output,
				self.targetActionMask: target_action_mask})
		return loss, Qmean_val

	def getWeights(self):
		return self.sess.run(self.weights)

	def getQValues(self, feed_dict):
		return self.sess.run(self.Qprime, feed_dict)

	def writeWeightsInFile(self, fileName):
		self.saver.save(self.sess, fileName)
