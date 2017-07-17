import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
from ..algorithms.cartPole_DQN import DQN
from ..models.FNN import FNN

env = gym.make('CartPole-v0')
ann = FNN(env)
dqn = DQN(env, ann)
dqn.train()
