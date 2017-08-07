from .sum_tree import SumTree
from ...parameters import HP
import math
import numpy as np

#TODO read about alpha beta values
class PriorizedStrategy:
	def __init__(self):
		self.memory =  SumTree(HP['size_of_experience'])

	def selectMiniBatch(self):
		return self.memory.sample(HP['mini_batch_size'])

#TODO think about below 100
	def storeExperience(self, exp):
		self.memory.insert(exp, 10)

	def getLength(self):
		return self.memory.getLength()

	def getHeldoutSet(self):
		temp, _, _ = self.selectMiniBatch()
		return [t['state'] for t in temp]

	def experienceReplay(self, model, target_weights, update_policy):
		experiences, probabilities, indexes = self.selectMiniBatch()
		X_val = []
		Y_val = []
		target_action_mask = np.zeros((len(experiences), model.output_size), dtype=int)
		weights = []
		states = []
		for index, exp in enumerate(experiences):
			X_val.append(exp['state'])
			target_action_mask[index][exp['action']] = 1
			states.append(exp['next_state'])
			w = lambda p: (p*HP['size_of_experience'])**(-1*HP['beta'])
			weights.append(w(probabilities[index]))
		nextStatesValues = update_policy.execute(model, states, target_weights)
		for index, exp in enumerate(experiences):
			if(exp['done']==True):
				Y_val.append(exp['reward'])
			else:
				Y_val.append(exp['reward'] + nextStatesValues[index]*HP['y'])
		weights /= max(weights)
		X_val = np.array(X_val)
		Y_val = np.array(Y_val)

		summary, TDerror  = model.executeStep(X_val, Y_val, target_action_mask, weights)
		#TODO vectorize this function
		for index, error in enumerate(TDerror):
			e = lambda err: (abs(err) + HP['epsilon_prio'])**HP['alpha']
			self.memory.update(indexes[index], e(error))
		return summary
