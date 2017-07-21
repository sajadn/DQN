import numpy as np
import random
from ..parameters import HP
from ..algorithms.DQNBase import DQNBase

#TODO seperate executing action from storing it
#TODO target nETWORK
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

    def executeAction(self, action, state):
        cumReward = 0
        cumDone = False
        states = []
        for i in range(HP['stacked_frame_size']):
            s1, reward, done, _ = self.env.step(action)
            states.append(s1)
            cumReward+= reward
            cumDone = cumDone or done
            if(cumDone == True):
                for j in range(i+1, HP['stacked_frame_size']):
                    states.append(s1)
                    break
        newState = self.model.preprocess(states)
        return {'state': state,
               'action': action,
               'reward': cumReward,
               'next_state': newState,
               'done': cumDone }
