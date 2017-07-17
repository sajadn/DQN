import gym
import numpy as np
import random
import tensorflow as tf
from ..parameters import HP
import matplotlib.pyplot as plt
from scipy.misc import imresize
from itertools import chain
from ..models.modelBase import modelBase


WIDTH = 84
class CNN(modelBase):
	def convertToGrayScale(self, frame):
		data = imresize(frame, (WIDTH, WIDTH, 3))
		return np.dot(data, [0.2126, 0.7152, 0.0722])

	def preprocess(self, input_val):
		f = np.vectorize(self.convertToGrayScale,signature='(n,m,k)->(h,i)')
		return np.dstack(tuple(f(input_val)))


	def definePlaceHolders(self):
		self.X = tf.placeholder(shape=[None, WIDTH, WIDTH, HP['stacked_frame_size']],dtype=tf.float32)
		self.Q = tf.placeholder(shape=[None],dtype=tf.float32)
		self.targetActionMask = tf.placeholder(shape=[None, self.output_size],dtype=tf.float32)

	def defineLossAndTrainer(self):
		temp = tf.reduce_sum(tf.multiply(self.Qprime, self.targetActionMask), 1)
		print ('temp shape', temp.shape)
		print ('Q shape ',self.Q.shape)
		self.loss = tf.reduce_mean(tf.square(temp -  self.Q))
		for weightRegul in self.weights[0::2]:
			self.loss += HP['regularization_factor'] * tf.reduce_sum(tf.square(weightRegul))
		trainer = tf.train.RMSPropOptimizer(
            		learning_rate = HP['learning_rate'],
           		epsilon=0.01,
            		decay=0.95,
            		momentum=0.95)
		self.step = trainer.minimize(self.loss)

	def defineNNArchitecture(self):
		#TODO add initiallization

		conv1 = tf.layers.conv2d(
      			inputs= self.X,
      			filters=32,
			strides=[4, 4],
      			kernel_size=[8, 8],
				name="c1",
      			activation=tf.nn.relu)
		conv2 = tf.layers.conv2d(inputs= conv1,
				filters=64,
				strides=[2, 2],
				name="c2",
      			kernel_size=[4, 4],
      			activation=tf.nn.relu)
		conv3 = tf.layers.conv2d(
  			inputs=conv2,
  			filters=64,
			name="c3",
  			kernel_size=[3, 3],
  			activation=tf.nn.relu)
		flatten = tf.reshape(conv3, [-1, 7 * 7 * 64])
		dense = tf.layers.dense(inputs=flatten, units=512, activation=tf.nn.relu)
		self.Qprime = tf.layers.dense(inputs=dense, units=self.output_size)
		self.P = tf.argmax(self.Qprime, 1)
		self.Qmean = tf.reduce_mean(self.Qprime)
		tfGraph = tf.get_default_graph()
		k = lambda x: tfGraph.get_tensor_by_name('c'+str(x)+'/kernel:0')
		b = lambda x: tfGraph.get_tensor_by_name('c'+str(x)+'/bias:0')
		self.weights = list(chain.from_iterable((k(x), b(x)) for x in range(1,4)))


	def executeStep(self, input_val, output, target_action_mask):
		_, loss, Qmean_val = self.sess.run([self.step, self.loss, self.Qmean],
			feed_dict = {
				self.X: input_val,
			 	self.Q: output,
				self.targetActionMask: target_action_mask})
		return loss, Qmean_val
