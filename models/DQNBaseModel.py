from ..models.modelBase import modelBase
import numpy as np
from ..parameters import HP
import tensorflow as tf
#TODO making HP and GAME_NAME global variable
import abc

class DQNBaseModel(modelBase):

	@abc.abstractmethod
	def defineInput(self):
		""
	def definePlaceHolders(self):
		self.X = self.defineInput()
		self.Q = tf.placeholder(shape=[None],dtype=tf.float32, name="true_values")
		self.targetActionMask = tf.placeholder(shape=[None, self.output_size],dtype=tf.float32, name="action_mask")
		self.prioritizedWeights = tf.placeholder(shape=[None], dtype=tf.float32, name="prioritizedWeights")
		self.reward = tf.placeholder(dtype=tf.float32, name="reward")
		tf.summary.scalar("reward", self.reward)

	def executeStep(self, input_val, output, target_action_mask, total_reward, prio_weights = np.ones(HP['mini_batch_size'])):
		_, summary, TDerror_val = self.sess.run([self.step, self.summary, self.TDerror],
			feed_dict = {
				self.X: input_val,
			 	self.Q: output,
				self.targetActionMask: target_action_mask,
				self.prioritizedWeights: prio_weights,
				self.reward: total_reward})
		return summary, TDerror_val

	@abc.abstractmethod
	def defineTrainer(self):
		""

	def defineLossAndTrainer(self):
		with tf.name_scope("loss"):
			temp = tf.multiply(tf.reduce_sum(tf.multiply(self.Qprime, self.targetActionMask), 1), self.prioritizedWeights)
			self.TDerror = temp -  self.Q
			#Huber function
			if(HP['error_clip']==1):
				self.loss = tf.reduce_sum(tf.where(tf.abs(self.TDerror)<1.0, 0.5*tf.square(self.TDerror), tf.abs(self.TDerror)-0.5))
			else:
				self.loss = tf.reduce_sum(tf.square(self.TDerror))
			tf.summary.scalar("loss", self.loss)
			for weightRegul in self.weights[0::2]:
				self.loss += (1/2)*HP['regularization_factor'] * tf.reduce_sum(tf.square(weightRegul))
		with tf.name_scope("trainer"):
			trainer = self.defineTrainer()
			self.step = trainer.minimize(self.loss)
