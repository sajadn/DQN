import numpy as np
from ...config import params
class doubleDQNStrategy:
    def execute(self, model, states, targetWeights):
        feed_dict = { model.X: states }
        Qvalues = model.getQValues(feed_dict)
        maxAction = np.argmax(Qvalues, axis = 1)
        feed_dict.update(zip(model.weights, targetWeights))
        Qvalues = model.getQValues(feed_dict)
        result = []
        for i in range(len(Qvalues)):
            result.append(Qvalues[i][maxAction[i]])
        return np.array(result)
