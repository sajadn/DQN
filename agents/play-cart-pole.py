import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from ast import literal_eval
from gym import wrappers
from ..models.FNN import FNN

loss = tf.reduce_mean(tf.reduce_sum(tf.square( Qprime -   Q), 1))
#  loss += HP['regular_factor'] * tf.reduce_sum(tf.square( weights[0::2]))
HP={'learning_rate': 0.00025}
trainer = tf.train.GradientDescentOptimizer(learning_rate = HP['learning_rate'])
step = trainer.minimize( loss)
saver = tf.train.Saver()


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
gameName = 'CartPole-v0'
env = gym.make(gameName)
saver.restore(sess, "extra/"+str(gameName)"/weights/model.ckpt")

sum = 0
# env = wrappers.Monitor(env, 'extra/CartPole-v0/monitor/experiment1',force=True)
for p in range(100):
	s = env.reset()
	for t in range(200):
		env.render()
		action = FFN.predictAction()
		s1, reward, done,info = env.step(action)
		s = s1
		if done == True:
			break
	print "episode", p , "score is: ",t+1
	sum += t+1
print sum/100
# env.close()
# gym.upload('monitor/cartpole-experiment-1', api_key='sk_YxX36rzIRtuinvJ0Pfmw')
