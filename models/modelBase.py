import abc
import tensorflow as tf
import os
from ..config import params


#TODO replace X with input and Q with output

class modelBase(abc.ABC):
    def __init__(self, env):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=params.GPU_fraction)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params.GPU_number)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.input_size = env.observation_space.shape[0]
        self.output_size = env.action_space.n - params.remove_no_op
        self.definePlaceHolders()
        self.defineNNArchitecture()
        self.defineLossAndTrainer()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.saver = tf.train.Saver()
        self.writer = tf.summary.FileWriter("DQN/extra/{}/tensorboard/{}".format(env.env.spec.id,params.folder_name))
        self.writer.add_graph(self.sess.graph)
        self.summary = tf.summary.merge_all()


    @abc.abstractmethod
    def definePlaceHolders(self):
        ""

    @abc.abstractmethod
    def defineNNArchitecture(self):
        ""

    @abc.abstractmethod
    def defineLossAndTrainer(self):
        ""

    @abc.abstractmethod
    def preprocess(self, input_val):
        ""

    def predictAction(self, input_v):
        return self.sess.run(self.P, feed_dict= { self.X : input_v })

    def getWeights(self):
        return self.sess.run(self.weights)

    def getQValues(self, feed_dict):
        return self.sess.run(self.Qprime, feed_dict)

    def writeWeightsInFile(self, fileName):
        self.saver.save(self.sess, fileName)

    def readFromFile(self, fileName):
        self.saver.restore(self.sess, fileName)
