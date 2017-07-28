import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
from ..parameters import HP
from ..algorithms.algorithmBase import algorithmBase
import abc


#DQN algorithm without error clipping
#TODO newtork target strategy typo
#TODO up to 30 no-op in the begining of each episode is required
class DQNBase(algorithmBase):

	def __init__(self, env, model, update_policy, memory_policy):
		self.env = env
		self.model = model
		self.memory_policy = memory_policy
		self.update_policy = update_policy
		self.GAME_NAME = self.env.env.spec.id
		self.total_steps = 0
		self.epsilon_decay = (HP['ep_start']-HP['ep_end'])/HP['ep_reduction_steps']

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
				action = self.env.action_space.sample()
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
		self.total_steps = 0
		total = 0.0
		for episode in range(1, HP['num_episodes']):
			state = self.initialState()
			while True:
				if(self.total_steps%HP['target_update'] == 0):
					self.target_weights = self.model.getWeights()
				action = self.selectAction(state)
				exp = self.executeAction(action, state)
				self.memory_policy.storeExperience(exp)
				self.total_steps += 1
				state = exp['next_state']
				total += exp['reward']
				summary = self.memory_policy.experienceReplay(
							self.model, self.target_weights, self.update_policy, total/50)
				if(HP['ep_start']>=(HP['ep_end']-0.00000001)):
					HP['ep_start'] -= self.epsilon_decay
				if(exp['done'] == True):
					break


			if(episode%50==0):
				print ('average (50E):', total/50)
				total = 0.0
				print ('e',HP['ep_start'])
				self.model.writer.add_summary(summary, episode)
				self.model.writeWeightsInFile(
					"Reinforcement-Learning/extra/{}/weights/model.ckpt".format(self.GAME_NAME))
			print ("Episode {} finished".format(episode))



	#e-greddy
	def selectAction(self, state):
		if np.random.rand(1) < HP['ep_start']:
			action = self.env.action_space.sample()
		else:
			action = self.model.predictAction([state])[0]
		return action



	def play(self):
		self.model.readFromFile(
		"Reinforcement-Learning/extra/{}/weights/model.ckpt".format(self.GAME_NAME))
		sum = 0
		for p in range(100):
			state = self.initialState()
			total = 0
			for _ in range(400):
				#self.env.render()
				a = np.random.rand(1)
				if(a<0.05):
					action = self.env.action_space.sample()
				else:
					action = self.model.predictAction([state])[0]
				exp = self.executeAction(action, state)
				state = exp['next_state']
				total+= exp['reward']
				if exp['done'] == True:
					break
			print ("episode", p , "score is: ",total)
			sum += total + 1
		print (sum/100)
