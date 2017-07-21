import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
from ..algorithms.cartPole_DQN import DQN
from ..models.FNN import FNN
from ..algorithms.update_strategy.targetNewtork import targetNetworkStrategy
import sys

env = gym.make('CartPole-v0')
ann = FNN(env)
tns = targetNetworkStrategy()
dqn = DQN(env, ann, tns)
arg = str(sys.argv[1])
if(arg == 'play'):
    dqn.play()
elif(arg == 'train'):
    dqn.train()
