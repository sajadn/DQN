import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from ast import literal_eval
from gym import wrappers

input_size = 4
output_size = 2
X = tf.placeholder(shape=[None, input_size],dtype=tf.float32)
Q = tf.placeholder(shape=[None, output_size],dtype=tf.float32)
W1 = tf.Variable(tf.random_uniform([ input_size, 128],0,0.01))
b1 = tf.Variable(tf.random_uniform([1, 128],0,0.01))
A1 = tf.add(tf.matmul(X, W1), b1)
Z1 = tf.nn.relu(A1)
W2 = tf.Variable(tf.random_uniform([128, 128],0,0.01))
b2 = tf.Variable(tf.random_uniform([1, 128],0,0.01))
A2 = tf.add(tf.matmul(Z1, W2), b2)
Z2 = tf.nn.relu(A2)
W3 = tf.Variable(tf.random_uniform([128,  output_size],0,0.01))
b3 = tf.Variable(tf.random_uniform([1,  output_size],0,0.01))
Qprime = tf.add(tf.matmul(Z2, W3), b3)
P = tf.argmax( Qprime, 1)
weights = [W1, b1, W2, b2, W3, b3]

loss = tf.reduce_mean(tf.reduce_sum(tf.square( Qprime -   Q), 1))
#  loss += HP['regular_factor'] * tf.reduce_sum(tf.square( weights[0::2]))
HP={'learning_rate': 0.00025}
trainer = tf.train.GradientDescentOptimizer(learning_rate = HP['learning_rate'])
step = trainer.minimize( loss)
saver = tf.train.Saver()


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver.restore(sess, "../model2.ckpt")

sum = 0
env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, '../monitor/cartpole-experiment-2',force=True)
for p in range(100):
	s = env.reset()
	for t in range(200):
		env.render()
		action = sess.run(P, feed_dict={X: [s]})[0]
		s1, reward, done,info = env.step(action)
		s = s1
		if done == True:
			break
	print "episode", p , "score is: ",t+1
	sum += t+1
print sum/100
# env.close()
# gym.upload('monitor/cartpole-experiment-1', api_key='sk_YxX36rzIRtuinvJ0Pfmw')
