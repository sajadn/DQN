import numpy as np
from ...parameters import HP
class targetNetworkStrategy:
    def execute(self, model, states, targetWeights):
        feed_dict = { model.X: states }
        feed_dict.update(zip(model.weights, targetWeights))
        Qvalues = model.getQValues(feed_dict)
        return np.max(Qvalues, axis = 1)
