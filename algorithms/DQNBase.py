import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
from ..parameters import HP
from ..algorithms.algorithmBase import algorithmBase
import tensorflow as tf
import abc


#DQN algorithm without error clipping
#TODO up to 30 no-op in the begining of each episode is required if change environment to ALE
class DQNBase(algorithmBase):

	def __init__(self, env, model, update_policy, memory_policy):
		self.env = env
		self.model = model
		self.memory_policy = memory_policy
		self.update_policy = update_policy
		self.GAME_NAME = self.env.env.spec.id
		self.total_steps = 0
		self.epsilon_first_decay = (HP['ep_start']-HP['ep_end'])/HP['ep_first_reduction']
		self.epsilon_second_decay = (HP['ep_end']-HP['ep_last'])/HP['ep_second_reduction']
		self.reward_tensor = tf.placeholder(shape=(),dtype=tf.float32, name="totalReward")
		self.reward_summ = tf.summary.scalar("reward", self.reward_tensor)
		self.epsilon_tensor = tf.placeholder(shape=(),dtype=tf.float32, name="totalReward")
		self.epsilon_summ = tf.summary.scalar("epsilon", self.epsilon_tensor)
		self.betaStep = (1-HP['beta'])/HP['ep_second_reduction']



	@abc.abstractmethod
	def initialState(self):
		""
	@abc.abstractmethod
	def executeAction(self, action):
		""

	def fillERMemory(self):
		while True:
			state = self.initialState()
			total = 0
			while True:
				action = self.selectAction(state, 1)
				exp = self.executeAction(action, state)
				self.memory_policy.storeExperience(exp)
				state = exp['next_state']
				total += exp['reward']
				if exp['done']:
					print("Episode finished")
					break
			if(self.memory_policy.getLength() >= (HP['initial_experience_sizes'])):
				break;
		print ("Random Agent Finished")



	def train(self):
		self.fillERMemory()
		self.heldout_set = self.memory_policy.getHeldoutSet()
		self.total_steps = 0
		total = 0.0
		for episode in range(1, HP['num_episodes']):
			state = self.initialState()
			# for _ in range(random.randint(0,30)):
			# 	state = self.executeAction(0, state)['state']
			while True:
				if(self.total_steps%HP['target_update'] == 0):
					self.target_weights = self.model.getWeights()
				action = self.selectAction(state, HP['ep_start'])
				exp = self.executeAction(action, state)
				self.memory_policy.storeExperience(exp)
				self.total_steps += 1
				state = exp['next_state']
				total += exp['reward']
				if(self.total_steps%HP['train_freq']==0):
					lossSummary = self.memory_policy.experienceReplay(self.model, self.target_weights,self.update_policy)
				if(HP['ep_start']>=HP['ep_end']):
					HP['ep_start'] -= self.epsilon_first_decay
				elif(HP['ep_start']>=HP['ep_last']):
					HP['ep_start'] -= self.epsilon_second_decay
				HP['beta'] += self.betaStep
				if(exp['done'] == True):
					break
				if(self.total_steps>=HP['max_step']):
					print ("Train Finished")
					return


			if(episode%50==0):
				print ('average (50E):', total/50)
				print ('step', self.total_steps)
				print ('e',HP['ep_start'])
				self.model.writer.add_summary(lossSummary, episode)
				qmeans = self.model.sess.run(self.model.QmeanSummary, feed_dict={self.model.X: self.heldout_set})
				self.model.writer.add_summary(qmeans, episode)
				self.model.writer.add_summary(self.model.sess.run(self.reward_summ, feed_dict={self.reward_tensor: total/50}), episode)
				self.model.writer.add_summary(self.model.sess.run(self.epsilon_summ, feed_dict={self.epsilon_tensor: HP['ep_start']}), episode)
				self.model.writeWeightsInFile(
					"Reinforcement-Learning/extra/{}/weights/{}/model.ckpt".format(self.GAME_NAME, HP['folder_name']))
				total = 0.0
			print ("Episode {} finished".format(episode))


	#e-greddy
	def selectAction(self, state, prob):
		if np.random.rand(1) <= prob:
			action = random.randint(0, self.model.output_size-1)
		else:
			action = self.model.predictAction([state])[0]
		return action



	def play(self):
		self.model.readFromFile(
		"Reinforcement-Learning/extra/{}/weights/{}/model.ckpt".format(self.GAME_NAME, HP['folder_name']))
		sum = 0
		for p in range(100):
			state = self.initialState()
			total = 0
			while True:
				self.env.render()
				action = self.selectAction(state, 0.01)
				exp = self.executeAction(action, state)
				state = exp['next_state']
				total+= exp['reward']
				if exp['done'] == True:
					break
			print ("episode", p , "score is: ",total)
			sum += total + 1
		print (sum/100)
