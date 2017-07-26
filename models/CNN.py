import gym
import numpy as np
import random
import tensorflow as tf
from ..parameters import HP
# import matplotlib.pyplot as plt
from scipy.misc import imresize, imsave
from itertools import chain
from ..models.modelBase import modelBase
from ..models.DQNBaseModel import DQNBaseModel



WIDTH = 84
class CNN(DQNBaseModel):

	def preprocess(self, frame):
		#TODO change order of these two lines
		data = np.dot(frame, [0.2126, 0.7152, 0.0722]).astype(np.uint8)
		return imresize(data, (WIDTH, WIDTH, 3))

	def stack_frames(self, frames):
		return np.dstack(tuple(frames))

	def defineInput(self):
	 	return tf.placeholder(shape=[None, WIDTH, WIDTH, HP['stacked_frame_size']],dtype=tf.float32, name="X")

	def defineTrainer(self):
		return tf.train.RMSPropOptimizer(
            		learning_rate = HP['learning_rate'],
           			epsilon=0.01,
            		decay=0.95,
            		momentum=0.95)

	def defineNNArchitecture(self):
		#TODO add initiallization
		initializer = tf.truncated_normal_initializer(0, 0.02)
		conv1 = tf.layers.conv2d(
				inputs= self.X,
				filters=32,
				strides=[4, 4],
				kernel_size=[8, 8],
				name="L1",
				kernel_initializer= initializer,
				activation=tf.nn.relu)

		conv2 = tf.layers.conv2d(
				inputs= conv1,
				filters=64,
				strides=[2, 2],
				name="L2",
				kernel_size=[4, 4],
				kernel_initializer= initializer,
				activation=tf.nn.relu)

		conv3 = tf.layers.conv2d(
				inputs=conv2,
				filters=64,
				name="L3",
				kernel_size=[3, 3],
				kernel_initializer= initializer,
				activation=tf.nn.relu)

		flatten = tf.reshape(conv3, [-1, 7 * 7 * 64])
		dense = tf.layers.dense(inputs=flatten, units=512, kernel_initializer=initializer, activation=tf.nn.relu, name="L4")
		self.Qprime = tf.layers.dense(inputs=dense, units=self.output_size, kernel_initializer=initializer, name="L5")
		self.P = tf.argmax(self.Qprime, 1)
		self.Qmean = tf.reduce_mean(self.Qprime)
		tf.summary.scalar("Qmean", self.Qmean)
		tfGraph = tf.get_default_graph()
		k = lambda x: tfGraph.get_tensor_by_name('L{}/kernel:0'.format(x))
		b = lambda x: tfGraph.get_tensor_by_name('L{}/bias:0'.format(x))
		self.weights = list(chain.from_iterable((k(x), b(x)) for x in range(1,6)))
