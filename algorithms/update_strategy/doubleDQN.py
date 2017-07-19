import numpy as np

class doubleDQNStrategy:
    def execute(self, model, exp, targetWeights):
        feed_dict = { model.X: [exp['next_state']] }
        Qvalues = model.getQValues(feed_dict)
        maxAction = np.argmax(Qvalues)
        feed_dict.update(zip(model.weights, targetWeights))
        Qvalues = model.getQValues(feed_dict)
        maxQ = Qvalues[maxAction]
        if(exp['done']==True):
            updatedValue = exp['reward']
        else:
            updatedValue = exp['reward'] + HP['y'] * maxQ
        return updatedValue
