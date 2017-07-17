import abc

class algorithmBase(abc.ABC):
    @abc.abstractmethod
    def train(self):
        ""

    @abc.abstractmethod
    def __init__(self, env, model):
        ""
