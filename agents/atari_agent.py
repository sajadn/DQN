import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
from ..algorithms.atari_DQN import DQN
from ..models.CNN import CNN
from ..algorithms.update_strategy.targetNetwork import targetNetworkStrategy
from ..algorithms.update_strategy.doubleDQN import doubleDQNStrategy
from ..parameters import HP
from ..algorithms.memory_strategy.normalStrategy import NormalStrategy
from ..algorithms.memory_strategy.prioritizedStrategy import PriorizedStrategy

import sys
from gym.envs.registration import registry, register



env = gym.make(str(sys.argv[2]))
ann = CNN(env)
upd = targetNetworkStrategy()
if(HP['doubleDQN']==1):
    upd = doubleDQNStrategy()
mem = NormalStrategy()
if(HP['prioritized']==1):
    mem = PriorizedStrategy()
dqn = DQN(env, ann, upd, mem)
arg = str(sys.argv[1])
if(arg == 'play'):
    dqn.play()
elif(arg == 'train'):
    dqn.train()
