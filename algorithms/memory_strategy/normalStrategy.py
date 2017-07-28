from ...parameters import HP
from collections import deque
import random
import numpy as np
from ..update_strategy.normalDQN import normalStrategy

class NormalStrategy:
	def __init__(self, ):
		self.memory = deque()
		self.noram = normalStrategy()

	def getLength(self):
		return len(self.memory)

	def selectMiniBatch(self):
		rcount = min(len(self.memory), HP['mini_batch_size'])
		return random.sample(self.memory, rcount)

	def storeExperience(self, exp):
		if(len(self.memory) > HP['size_of_experience']):
			self.memory.popleft()
		self.memory.append(exp)

	def experienceReplay(self, model, target_weights, update_policy, total_reward):
		sexperiences = self.selectMiniBatch()
		X_val = []
		Y_val = []
		# Y_fuck = []
		next_states = []
		target_action_mask = np.zeros((len(sexperiences), model.output_size), dtype=int)
		for index, exp in enumerate(sexperiences):
			X_val.append(exp['state'])
			target_action_mask[index][exp['action']] = 1
			next_states.append(exp['next_state'])
		temps = update_policy.execute(model, next_states, target_weights)
		# fucks = self.noram.execute(model, next_states, target_weights)
		for index, exp in enumerate(sexperiences):
			if(exp['done']==True):
				Y_val.append(exp['reward'])
				# Y_fuck.append(exp['reward'])
			else:
				Y_val.append(exp['reward'] + temps[index]*HP['y'])
				# Y_fuck.append(exp['reward']+fucks[index])
		# feed_dict = { model.X: X_val }
		# XQvalues = model.getQValues(feed_dict)
		# for index in range(len(XQvalues)):
		# 	print ('current', XQvalues[index][exp['action']])
		# 	print ('next', Y_val[index])
		# 	print ('fuck next', Y_fuck[index])
		# 	print ('**')
		X_val = np.array(X_val)
		Y_val = np.array(Y_val)
		summary, TDerror = model.executeStep(X_val, Y_val, target_action_mask, total_reward)
		# print ("************************ experience replay finished *************************")
		return summary
