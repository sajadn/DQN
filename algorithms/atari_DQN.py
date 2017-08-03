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
        s1, reward, done, _ = self.env.step(action)
        reward = reward if reward==0 else reward/abs(reward)
        newObservation = self.model.preprocess(s1)
        newState = state[:, :, 1:]
        newState = np.dstack((newState, newObservation))
        return {'state': state,
               'action': action,
               'reward': reward,
               'next_state': newState,
               'done': done }
