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
class DQNBase(algorithmBase):

	def __init__(self, env, model):
		self.env = env
		self.model = model
		self.expStore =  deque()
		self.total_steps = 0
		self.GAME_NAME = self.env.env.spec.id

	@abc.abstractmethod
	def initialState(self):
		""
	@abc.abstractmethod
	def executeActionStoreIt(self, action):
		""

	def fillERMemory(self):
		for i in range(HP['initial_experience_sizes']):
			self.s = self.initialState()
			total = 0
			while True:
				action = self.env.action_space.sample()
				done, reward = self.executeActionStoreIt(action)
				if done:
					print("Episode {} finished after {} timesteps".format(i,total))
					break

	def train(self):
		self.fillERMemory()
		self.total_steps = 0
		sum = 0
		for episode in range(HP['num_episodes']):
			self.s = self.initialState()
			t = 0
			while True:
				if(self.total_steps%HP['target_update'] == 0):
					self.target_weights = self.model.getWeights()
				action = self.selectAction()
				done, reward = self.executeActionStoreIt(action)
				t += reward
				self.total_steps += 1
				if(done == True):
					break
				l = self.experienceReplay()
				if(self.total_steps%250==0):
					print (l, HP['e'])
				if(HP['e']>=0.2):
					if(self.total_steps%HP['reducing_e_freq']==0):
						HP['e'] -= 0.1
			print ("Episode {} finished with score: {}".format(episode,t))
		self.model.writeWeightsInFile("extra/"+self.GAME_NAME+"/weights/model.ckpt")

	#e-greddy
	def selectAction(self):
		if np.random.rand(1) < HP['e']:
			action = self.env.action_space.sample()
		else:
			action = self.model.predictAction([self.s])[0]
			return action

	def selectMiniBatch(self):
		rcount = min(len(self.expStore), HP['mini_batch_size'])
		return random.sample(self.expStore,rcount)

	def experienceReplay(self):
		sexperiences = self.selectMiniBatch()
		X_val = []
		Y_val = []
		target_action_mask = np.zeros((len(sexperiences), self.model.output_size), dtype=int)
		for index, exp in enumerate(sexperiences):
			feed_dict = {self.model.X: [exp['next_state']]}
			feed_dict.update(zip(self.model.weights, self.target_weights))
			Q1val = self.model.getQValues(feed_dict)
			maxQ1 = np.max(Q1val)
			X_val.append(exp['state'])
			target_action_mask[index][exp['action']] = 1
			if(exp['done']==True):
				cQ = exp['reward']
			else:
				cQ = exp['reward'] + HP['y'] * maxQ1
			Y_val.append(cQ)
		X_val = np.array(X_val)
		Y_val = np.array(Y_val)
		return self.model.executeStep(X_val, Y_val, target_action_mask)
