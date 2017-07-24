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
		self.X = tf.placeholder(shape=[None, self.input_size],dtype=tf.float32, name="X")
		self.Q = tf.placeholder(shape=[None],dtype=tf.float32, name="true_values")
		self.targetActionMask = tf.placeholder(shape=[None, self.output_size],dtype=tf.float32, name="action_mask")
		self.prioritizedWeights = tf.placeholder(shape=[None], dtype=tf.float32, name="prioritizedWeights")
		self.reward = tf.placeholder(dtype=tf.float32, name="reward")
		tf.summary.scalar("reward", self.reward)



	def defineNNArchitecture(self):
		with tf.name_scope("layer1"):
			W1 = tf.Variable(tf.random_uniform([self.input_size, 128],0,0.01), name = "W")
			b1 = tf.Variable(tf.random_uniform([1, 128],0,0.01), name= "B")
			A1 = tf.add(tf.matmul(self.X, W1), b1, name= "A")
			Z1 = tf.nn.relu(A1, name= "Z")
			tf.summary.histogram("weights", W1)
			tf.summary.histogram("biases", b1)
		with tf.name_scope("layer2"):
			W2 = tf.Variable(tf.random_uniform([128, 128],0,0.01), name="W")
			b2 = tf.Variable(tf.random_uniform([1, 128],0,0.01), name="B")
			A2 = tf.add(tf.matmul(Z1, W2), b2, name="A")
			Z2 = tf.nn.relu(A2, name="Z")
			tf.summary.histogram("weights", W2)
			tf.summary.histogram("biases", b2)
		with tf.name_scope("outputLayer"):
			W3 = tf.Variable(tf.random_uniform([128, self.output_size],0,0.01), name="W")
			b3 = tf.Variable(tf.random_uniform([1, self.output_size],0,0.01), name="B")
			tf.summary.histogram("weights", W3)
			tf.summary.histogram("biases", b3)
			self.Qprime = tf.add(tf.matmul(Z2, W3), b3, name="output")
		self.P = tf.argmax(self.Qprime, 1, name="prectedAction")
		self.Qmean = tf.reduce_mean(self.Qprime)
		tf.summary.scalar("Qmean", self.Qmean)
		self.weights = [W1, b1, W2, b2, W3, b3]

	def defineLossAndTrainer(self):
		with tf.name_scope("loss"):
			temp = tf.multiply(tf.reduce_sum(tf.multiply(self.Qprime, self.targetActionMask), 1), self.prioritizedWeights, name="action_mask_times_actions")
			self.TDerror = tf.subtract(temp, self.Q, name="TD-error")
			if(HP['error_clip']==1):
				print ("with clipping")
				self.loss = tf.reduce_mean(tf.where(tf.abs(self.TDerror)<1, 0.5*tf.square(self.TDerror), tf.abs(self.TDerror)-0.5), name="loss_value")
			else:
				print ("without clipping")
				self.loss = tf.reduce_mean(tf.square(self.TDerror), name="loss_value")
			for weightRegul in self.weights[0::2]:
				self.loss += HP['regularization_factor'] * tf.reduce_sum(tf.square(weightRegul), name="regularization")
			tf.summary.scalar("loss", self.loss)
		with tf.name_scope("train"):
			trainer = tf.train.GradientDescentOptimizer(learning_rate = HP['learning_rate'])
			self.step = trainer.minimize(self.loss)

	def preprocess(self, input_val):
		return input_val

	def executeStep(self, input_val, output, target_action_mask, total_reward, weights = np.ones(HP['mini_batch_size'])):
		_, summary, TDerror_val = self.sess.run([self.step, self.summary, self.TDerror],
			feed_dict = {
				self.X: input_val,
			 	self.Q: output,
				self.targetActionMask: target_action_mask,
				self.prioritizedWeights: weights,
				self.reward: total_reward})
		return summary, TDerror_val
