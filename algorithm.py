import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
from parameters import HP


#DQN algorithm without error clipping
class DQN:

	def __init__(self, env, model):
		self.env = env
		self.model = model
		self.expStore =  deque()
		self.total_steps = 0
	def fillingExperienceReplayMemory(self):
		for _ in range(2000):
		    self.s = self.env.reset()
		    for t in range(200):
		        action = self.env.action_space.sample()
		        done = self.executeActionStoreIt(action)
		        if done:
		            print("Episode finished after {} timesteps".format(t+1))
		            break


	def train(self):
		self.total_steps = 0
		sum = 0
		for episode in range(HP['num_episodes']):
			self.s = self.env.reset()
			t = 0
			for t in range(HP['steps_per_episodes']):
				if(self.total_steps%HP['target_update'] == 0):
					self.target_weights = self.model.getWeights()
				action = self.selectAction()
				done = self.executeActionStoreIt(action)
				self.total_steps += 1
				if(done == True):
					break
				l = self.experienceReplay()
				if(self.total_steps%250==0):
					print l, HP['e']
				if(HP['e']>=0.2):
					if(self.total_steps%HP['reducing_e_freq']==0):
						HP['e'] -= 0.1
			sum += t
			print "episode ", episode, "finished with score: ", t
		# print "epoch ", epoch, "finished with average score: ",sum/HP['num_episodes']
		self.model.writeWeightsInFile("../model2.ckpt")

	#e-greddy
	def selectAction(self):
		if np.random.rand(1) < HP['e']:
			action = self.env.action_space.sample()
		else:
			action = self.model.predictAction([self.s])[0]
		return action


	def executeActionStoreIt(self, action):
		s1, reward, done, info = self.env.step(action)
		if(done == True):
			reward = -100
		exp = {'state': self.s,
				'action': action,
				'reward': reward,
				'next_state': s1,
				'done': done}
		if(len(self.expStore) > HP['size_of_experience']):
			self.expStore.popleft()
		self.expStore.append(exp)
		self.s = s1
		return done

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
