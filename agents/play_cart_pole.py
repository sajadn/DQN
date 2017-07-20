import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from ast import literal_eval
from gym import wrappers
from ..models.CNN import CNN
from ..algorithms.atari_DQN import DQN
from ..algorithms.update_strategy.targetNewtork import targetNetworkStrategy

# loss = tf.reduce_mean(tf.reduce_sum(tf.square( Qprime -   Q), 1))
#  loss += HP['regular_factor'] * tf.reduce_sum(tf.square( weights[0::2]))
# HP={'learning_rate': 0.00025}
# trainer = tf.train.GradientDescentOptimizer(learning_rate = HP['learning_rate'])
# step = trainer.minimize( loss)
# saver = tf.train.Saver()



#gameName = 'CartPole-v0'
#env = gym.make(gameName)
#mFNN = FNN(env)
#mFNN.readFromFile("Reinforcement-Learning/extra/{}/weights/model.ckpt".format(gameName	))

#sum = 0
# env = wrappers.Monitor(env, 'extra/CartPole-v0/monitor/experiment1',force=True)
#for p in range(100):
#	s = env.reset()
#	for t in range(200):
#		env.render()
#		action = mFNN.predictAction([s])
#		s1, reward, done,info = env.step(action[0])
#		s = s1
#		if done == True:
#			break
#	print ("episode", p , "score is: ",t+1)
#	sum += t+1
#print (sum/100)
# env.close()
#gym.upload('monitor/cartpole-experiment-1', api_key='sk_YxX36rzIRtuinvJ0Pfmw')


gameName = 'Breakout-v0'
env = gym.make(gameName)
mCNN = CNN(env)
tns = targetNetworkStrategy()
dqn = DQN(env, mCNN, tns)

mCNN.readFromFile("Reinforcement-Learning/extra/{}/weights/model.ckpt".format(gameName))

sum = 0
#env = wrappers.Monitor(env, 'extra/CartPole-v0/monitor/experiment1',force=True)
for p in range(100):
       s = dqn.initialState()
       print ('7878', s.shape)
       total = 0
       for t in range(200):
#               env.render()
               action = mCNN.predictAction([s])
               print (np.array([s]).shape)
               done, reward, s1 = dqn.executeActionStoreIt(action[0])
               s = s1
               total+= reward
               if done == True:
                       break
       print ("episode", p , "score is: ",tota)
       sum += t+1
print (sum/100)

