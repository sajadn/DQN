import numpy as np


class normalStrategy:
    def execute(self, model, exp, targetWeights):
        #target weights isn't used
        feed_dict = { model.X: [exp['next_state']] }
        Qvalues = model.getQValues(feed_dict)
        maxQ = np.max(Qvalues)
        if(exp['done']==True):
            updatedValue = exp['reward']
        else:
            updatedValue = exp['reward'] + HP['y'] * maxQ
        return updatedValue
