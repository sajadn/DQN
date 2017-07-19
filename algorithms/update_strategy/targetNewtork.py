import numpy as np
from ...parameters import HP
class targetNetworkStrategy:
    def execute(self, model, exp, targetWeights):
        feed_dict = { model.X: [exp['next_state']] }
        feed_dict.update(zip(model.weights, targetWeights))
        Qvalues = model.getQValues(feed_dict)
        maxQ = np.max(Qvalues)
        if(exp['done']==True):
            updatedValue = exp['reward']
        else:
            updatedValue = exp['reward'] + HP['y'] * maxQ
        return updatedValue
