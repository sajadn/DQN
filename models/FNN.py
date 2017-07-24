import gym
import numpy as np
import random
import tensorflow as tf
from ..parameters import HP
from ..models.modelBase import modelBase
from ..models.DQNBaseModel import DQNBaseModel


#TODO DQNBase model
# a neural network with two hidden layers (128 nodes for each)

class FNN(DQNBaseModel):

	def defineInput(self):
		return tf.placeholder(shape=[None, self.input_size],dtype=tf.float32, name="X")
		
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

	def defineTrainer(self):
		return tf.train.GradientDescentOptimizer(learning_rate = HP['learning_rate'])

	def preprocess(self, input_val):
		return input_val
