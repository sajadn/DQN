import numpy as np
import random
from ..parameters import HP
from ..algorithms.DQNBase import DQNBase

#TODO seperate executing action from storing it
class DQN(DQNBase):
    def initialState(self):
        states = []
        action = self.env.action_space.sample()
        obs = self.env.reset()
        states.append(obs)
        for _ in range(HP['stacked_frame_size']-1):
            obs, reward, done, info = self.env.step(action)
            states.append(obs)
        return self.model.preprocess(states)

    def executeActionStoreIt(self, action):
        cumReward = 0
        cumDone = False
        states = []
        for _ in range(HP['stacked_frame_size']):
            s1, reward, done, _ = self.env.step(action)
            states.append(s1)
            cumReward+= reward
            cumDone = cumDone or done
            if(cumDone == True):
                return cumDone, cumReward
        newState = self.model.preprocess(states)
        exp = {'state': self.s,
               'action': action,
               'reward': cumReward,
               'next_state': newState,
               'done': cumDone}
        if(len(self.expStore) > HP['size_of_experience']):
            self.expStore.popleft()
        self.expStore.append(exp)
        self.s = newState
        return cumDone, cumReward
