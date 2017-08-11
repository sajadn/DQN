import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
from ..config import params
from ..algorithms.algorithmBase import algorithmBase
import tensorflow as tf
import abc
import os


#TODO up to 30 no-op in the begining of each episode is required if change environment to ALE
class DQNBase(algorithmBase):

	def __init__(self, env, model, update_policy, memory_policy):
		self.env = env
		self.model = model
		self.memory_policy = memory_policy
		self.update_policy = update_policy
		self.GAME_NAME = self.env.env.spec.id
		self.total_steps = 0
		self.epsilon_first_decay = (params.epsilon-params.epsilon_end)/params.ep_first_reduction
		self.epsilon_second_decay = (params.epsilon_end-params.epsilon_last)/params.ep_second_reduction
		self.reward_tensor = tf.placeholder(shape=(),dtype=tf.float32, name="totalReward")
		self.reward_summ = tf.summary.scalar("reward", self.reward_tensor)
		self.epsilon_tensor = tf.placeholder(shape=(),dtype=tf.float32, name="epsilon")
		self.epsilon_summ = tf.summary.scalar("epsilon", self.epsilon_tensor)
		self.betaStep = (1-params.beta)/params.ep_second_reduction
		self.variablesDirectory = "DQN/extra/{}/weights/{}/model.ckpt".format(self.GAME_NAME, params.folder_name)




	@abc.abstractmethod
	def initialState(self):
		""
	@abc.abstractmethod
	def executeAction(self, action):
		""

	def train(self):
		total_reward = 0.0
		state = self.initialState()
		self.target_weights = self.model.getWeights()
		# for _ in range(random.randint(0,30)):
		# 	state = self.executeAction(0, state)['state']
		self.heldout_set = None
		episode = 1
		for total_steps in range(params.max_step):
			action = self.selectAction(state, params.epsilon)
			exp = self.executeAction(action, state)
			self.memory_policy.storeExperience(exp)
			state = exp['next_state']
			if(total_steps > params.initial_experience_sizes):
				total_reward += exp['reward']
				if(self.heldout_set == None):
					self.heldout_set = self.memory_policy.getHeldoutSet()
				if(total_steps % params.target_update == 0):
					self.target_weights = self.model.getWeights()
				if(total_steps % params.train_frequency == 0):
					lossSummary = self.memory_policy.experienceReplay(
					self.model, self.target_weights,self.update_policy)
				if(params.epsilon >= params.epsilon_end):
					params.epsilon -= self.epsilon_first_decay
				elif(params.epsilon>=params.epsilon_last):
					params.epsilon -= self.epsilon_second_decay
				params.beta += self.betaStep
			if(exp['done'] == True):
				state = self.env.reset()
				if(total_steps > params.initial_experience_sizes):
					print ("Episode {} finished".format(episode))
					episode+=1
					if(episode%51==0 ):
						self.writeSummary(total_steps, total_reward,
						 lossSummary, episode)
						total_reward = 0.0



	#e-greddy
	def selectAction(self, state, prob):
		if np.random.rand(1) <= prob:
			action = random.randint(0, self.model.output_size-1)
		else:
			action = self.model.predictAction([state])[0]
		return action

	def writeSummary(self, total_steps, total_reward, lossSummary, episode):
		print ('average (50E):', total_reward/50)
		print ('step', total_steps)
		print ('e',params.epsilon)
		self.model.writer.add_summary(lossSummary, episode)
		qmeans = self.model.sess.run(self.model.QmeanSummary,
		feed_dict={self.model.X: self.heldout_set})
		self.model.writer.add_summary(qmeans, episode)
		self.model.writer.add_summary(self.model.sess.run(self.reward_summ,
		 feed_dict={self.reward_tensor: total_reward/50}), episode)
		self.model.writer.add_summary(self.model.sess.run(self.epsilon_summ,
		 feed_dict={self.epsilon_tensor: params.epsilon}), episode)
		if not os.path.exists(self.variablesDirectory):
		    os.makedirs(self.variablesDirectory)
		self.model.writeWeightsInFile(self.variablesDirectory)






	def play(self):
		self.model.readFromFile(self.variablesDirectory)
		sum = 0
		for p in range(100):
			state = self.initialState()
			total = 0
			while True:
				self.env.render()
				action = self.selectAction(state, 1)
				exp = self.executeAction(action, state)
				state = exp['next_state']
				total+= exp['reward']
				if exp['done'] == True:
					break
			print ("episode", p , "score is: ",total)
			sum += total + 1
		print (sum/100)
