from ...parameters import HP
from collections import deque
import random
import numpy as np

class NormalStrategy:
	def __init__(self, ):
		self.memory = deque()

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
		next_states = []
		target_action_mask = np.zeros((len(sexperiences), model.output_size), dtype=int)
		for index, exp in enumerate(sexperiences):
			X_val.append(exp['state'])
			target_action_mask[index][exp['action']] = 1
			next_states.append(exp['next_state'])
		temps = update_policy.execute(model, next_states, target_weights)
		for index, exp in enumerate(sexperiences):
			if(exp['done']==True):
				Y_val.append(exp['reward'])
			else:
				Y_val.append(exp['reward'] + temps[index]*HP['y'])
		X_val = np.array(X_val)
		Y_val = np.array(Y_val)
		summary, TDerror = model.executeStep(X_val, Y_val, target_action_mask, total_reward)
		return summary
