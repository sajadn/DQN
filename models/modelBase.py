import abc
import tensorflow as tf
import os


#TODO replace X with input and Q with output

#TODO search about how to force subclasses to have concrete fields
class modelBase(abc.ABC):
    def __init__(self, env):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.sess = tf.Session()
        self.input_size = env.observation_space.shape[0]
        self.output_size = env.action_space.n
        self.definePlaceHolders()
        self.defineNNArchitecture()
        self.defineLossAndTrainer()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.saver = tf.train.Saver()
        self.writer = tf.summary.FileWriter("Reinforcement-Learning/extra/{}/tensorboard/15".format(env.env.spec.id))
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
