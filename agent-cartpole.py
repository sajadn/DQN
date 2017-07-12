import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
from model import NN
from algorithm import DQN

env = gym.make('CartPole-v0')
ann = NN(env)
dqn = DQN(env, ann)
dqn.fillingExperienceReplayMemory()
dqn.train()
