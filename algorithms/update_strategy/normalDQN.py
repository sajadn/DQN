import numpy as np


class normalStrategy:
    def execute(self, model, states, targetWeights):
        #target weights isn't used
        feed_dict = { model.X: states }
        Qvalues = model.getQValues(feed_dict)
        return np.max(Qvalues, axis = 1)
