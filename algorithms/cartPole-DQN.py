import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
from ..parameters import HP
from ..algorithms.DQNBase import DQNBase


#DQN algorithm without error clipping
class DQNBase(DQNBase):

	def initialState(self):
		return self.env.reset()

    def executeActionStoreIt(self, action):
		s1, reward, done, info = self.env.step(action)
		storeReward = reward if (done==False) else -100
		exp = {'state': self.s,
				'action': action,
				'reward': storeReward,
				'next_state': s1,
				'done': done}
		if(len(self.expStore) > HP['size_of_experience']):
			self.expStore.popleft()
		self.expStore.append(exp)
		self.s = s1
		print "basic",done,reward
		return done, reward
