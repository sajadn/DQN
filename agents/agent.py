import gym
import sys
from ..models.CNN import CNN
from ..models.FNN import FNN
from ..algorithms.atari_DQN import DQN as aDQN
from ..algorithms.classic_DQN import DQN as cDQN
from ..algorithms.update_strategy.targetNetwork import targetNetworkStrategy
from ..algorithms.update_strategy.doubleDQN import doubleDQNStrategy
from ..algorithms.memory_strategy.normalStrategy import NormalStrategy
from ..algorithms.memory_strategy.prioritizedStrategy import PriorizedStrategy
from ..config import params

def run():
    env = gym.make(params.envName)
    if(params.doubleDQN==1):
        upd = doubleDQNStrategy()
    else:
        upd = targetNetworkStrategy()
    if(params.prioritized_experience==1):
        mem = PriorizedStrategy()
    else:
        mem = NormalStrategy()
    if(params.envType=="atari"):
        model = CNN(env)
        dqn = aDQN(env, model, upd, mem)
    elif(params.envType=="classic"):
        model = FNN(env)
        dqn = cDQN(env, model, upd, mem)
    if(params.action == 'play'):
        dqn.play()
    elif(params.action == 'train'):
        dqn.train()
