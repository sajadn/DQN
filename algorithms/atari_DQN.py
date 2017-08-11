import numpy as np
import random
from ..config import params
from ..algorithms.DQNBase import DQNBase

class DQN(DQNBase):
    def initialState(self):
        states = []
        obs = self.env.reset()
        obs = self.model.preprocess(obs)
        for _ in range(params.stacked_frame_size):
            states.append(obs)
        return np.dstack(tuple(states))

    def executeAction(self, action, state):
        cumReward = 0
        pf = []
        for i in range(params.frame_skipping):
            s1, reward, done, _ = self.env.step(action+ params.remove_no_op)
            clip = lambda r: r if r==0 else r/abs(r)
            cumReward += clip(reward)
            if(done == True):
                break
            if(i != (params.frame_skipping-1)):
                pf = s1

        if(len(pf) != 0):
            s1 = np.maximum(s1, pf)

        newState = self.appendNewObservation(s1, state)
        return {'state': state,
               'action': action,
               'reward': cumReward,
               'next_state': newState,
               'done': done }

    def appendNewObservation(self, observation, state):
        observation = self.model.preprocess(observation)
        newState = state[:, :, 1:]
        return np.dstack((newState, observation))
