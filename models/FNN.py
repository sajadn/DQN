import gym
import numpy as np
import random
import tensorflow as tf
from ..parameters import HP
from ..models.modelBase import modelBase


#TODO DQNBase model

# a neural network with two hidden layers (128 nodes for each)
class FNN(modelBase):

	def definePlaceHolders(self):
		self.X = tf.placeholder(shape=[None, self.input_size],dtype=tf.float32)
		self.Q = tf.placeholder(shape=[None],dtype=tf.float32)
		self.targetActionMask = tf.placeholder(shape=[None, self.output_size],dtype=tf.float32)
		self.prioritizedWeights = tf.placeholder(shape=[None], dtype=tf.float32)


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
		temp = tf.multiply(tf.reduce_sum(tf.multiply(self.Qprime, self.targetActionMask), 1), self.prioritizedWeights)
		print ('temp shape', temp.shape)
		print ('Q shape ',self.Q.shape)
		self.TDerror = temp -  self.Q
		if(HP['error_clip']==1):
			print ("with clipping")
			self.loss = tf.reduce_mean(tf.where(tf.abs(self.TDerror)<0.5, tf.square(self.TDerror), tf.abs(self.TDerror)))
		else:
			print ("without clipping")
			self.loss = tf.reduce_mean(tf.square(self.TDerror))
		for weightRegul in self.weights[0::2]:
			self.loss += HP['regularization_factor'] * tf.reduce_sum(tf.square(weightRegul))
		trainer = tf.train.GradientDescentOptimizer(learning_rate = HP['learning_rate'])
		self.step = trainer.minimize(self.loss)

	def preprocess(self, input_val):
		return input_val

	def executeStep(self, input_val, output, target_action_mask, weights = np.ones(HP['mini_batch_size'])):
		_, loss, Qmean_val, TDerror_val = self.sess.run([self.step, self.loss, self.Qmean, self.TDerror],
			feed_dict = {
				self.X: input_val,
			 	self.Q: output,
				self.targetActionMask: target_action_mask,
				self.prioritizedWeights: weights})
		return loss, Qmean_val, TDerror_val
