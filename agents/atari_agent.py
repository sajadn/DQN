import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
from ..algorithms.atari_DQN import DQN
from ..models.CNN import CNN
from ..algorithms.update_strategy.targetNewtork import targetNetworkStrategy
import sys

env = gym.make('Breakout-v0')
ann = CNN(env)
tns = targetNetworkStrategy()
dqn = DQN(env, ann, tns)
arg = str(sys.argv[1])
if(arg == 'play'):
    dqn.play()
elif(arg == 'train'):
    dqn.train()
