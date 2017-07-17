import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
from ..algorithms.atari_DQN import DQN
from ..models.CNN import CNN

env = gym.make('Breakout-v0')
ann = CNN(env)
dqn = DQN(env, ann)
dqn.train()
