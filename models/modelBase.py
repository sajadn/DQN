import abc
import tensorflow as tf

#TODO replace X with input and Q with output

#TODO search about how to force subclasses to have concrete fields
class modelBase(abc.ABC):
    def __init__(self, env):
        self.sess = tf.Session()
        self.input_size = env.observation_space.shape[0]
        self.output_size = env.action_space.n
        self.definePlaceHolders()
        self.defineNNArchitecture()
        self.defineLossAndTrainer()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.saver = tf.train.Saver()

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
