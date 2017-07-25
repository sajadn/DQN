import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
from ..parameters import HP
from ..algorithms.DQNBase import DQNBase


#DQN algorithm without error clipping
class DQN(DQNBase):

	def initialState(self):
		return self.env.reset()

	def executeAction(self, action, state):
		s1, reward, done, info = self.env.step(action)
		# storeReward = reward if (done==False) else -1
		return {'state': state,
				'action': action,
				'reward': reward,
				'next_state': s1,
				'done': done}
