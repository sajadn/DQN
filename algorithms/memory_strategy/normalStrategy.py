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

	def experienceReplay(self, model, target_weights, update_policy):
		sexperiences = self.selectMiniBatch()
		X_val = []
		Y_val = []
		target_action_mask = np.zeros((len(sexperiences), model.output_size), dtype=int)
		for index, exp in enumerate(sexperiences):
			X_val.append(exp['state'])
			target_action_mask[index][exp['action']] = 1
			Y_val.append(update_policy.execute(model, exp, target_weights))
		X_val = np.array(X_val)
		Y_val = np.array(Y_val)
		*returnValue, TDerror = model.executeStep(X_val, Y_val, target_action_mask)
		return returnValue
