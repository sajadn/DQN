import numpy as np
import random
from ..parameters import HP
from ..algorithms.DQNBase import DQNBase

class DQN(DQNBase):
    def initialState(self):
        states = []
        obs = self.env.reset()
        obs = self.model.preprocess(obs)
        for _ in range(HP['stacked_frame_size']):
            states.append(obs)
        return np.dstack(tuple(states))

    def executeAction(self, action, state):
        cumReward = 0
        pf = []
        for i in range(HP['frame_skipping']):
            s1, reward, done, _ = self.env.step(action)
            clip = lambda r: r if r==0 else r/abs(r)
            cumReward += clip(reward)
            if(done == True):
                break
            if(i != (HP['frame_skipping']-1)):
                pf = s1

        if(len(pf) != 0):
            s1 = np.maximum(s1, pf)
        newObservation = self.model.preprocess(s1)
        newState = state[:, :, 1:]
        newState = np.dstack((newState, newObservation))
        return {'state': state,
               'action': action,
               'reward': cumReward,
               'next_state': newState,
               'done': done }
