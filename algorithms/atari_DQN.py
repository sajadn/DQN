import numpy as np
import random
from ..parameters import HP
from ..algorithms.DQNBase import DQNBase

#TODO seperate executing action from storing it
#TODO target nETWORK
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
        for _ in range(HP['frame_skipping']):
            s1, reward, done, _ = self.env.step(action)
            cumReward += reward
            if done==True:
                break
        newObservation = self.model.preprocess(s1)
        newState = state[:, :, 1:]
        newState = np.dstack((newState, newObservation))
        return {'state': state,
               'action': action,
               'reward': cumReward,
               'next_state': newState,
               'done': done }
