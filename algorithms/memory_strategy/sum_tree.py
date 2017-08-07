import math
import numpy as np
import random

class SumTree:

    def __init__(self, size, values=None, probs=None):
        self.size = size
        self.tree_levels = int(math.ceil(math.log(size, 2))+1)
        self.data = np.zeros(2**(self.tree_levels-1), dtype= np.object)
        self.prob = np.zeros((2**self.tree_levels)-1)
        self.start = int(2**(self.tree_levels-1))-1
        self._cursor = self.start
        self.flag = False
        if values is not None and probs is not None:
            self.initial_values(values, probs)

    def getLength(self):
        if(self.flag == True):
            return size
        else:
            return self._cursor-self.start

    def initial_values(self, values, probs):
        for index in range(len(values)):
            self.prob[self.start + index] = probs[index]
            self.data[index] = values[index]
        self._cursor = self.start+index+1
        for level in range(self.tree_levels - 1, 0, -1):
            for i in range(2**(level-1)-1, 2**(level)-1, +1):
                childs = self._child(i)
                self.prob[i] = self._sum_children(i)

    def insert(self, value, probs):
        if(self._cursor> 2**(self.tree_levels)-2):
            self._cursor= self.start
            self.flag = True
        self.data[self._cursor - self.start] = value
        self.prob[self._cursor] = probs
        self._update(self._parent(self._cursor))
        self._cursor += 1


    def update(self, index, prob):
        self.prob[index] = prob
        self._update(self._parent(index))

#TODO what will happen if size is more than sum-tree values
    def sample(self, size):
        values = []
        indexes = []
        probs = []
        randoms = random.sample(range(int(self.prob[0])), size)
        for rn in randoms:
            index = self._sample(rn, 0)
            if index not in indexes:
                values.append(self.data[index - self.start])
                probs.append(self.prob[index])
                indexes.append(index)

        return values, probs, indexes

    def _parent(self, index):
        return int(index/2)-1 if(index%2==0) else int((index-1)/2)

    def _child(self, index):
        return int(index*2+1), int(index*2+2)

    def _sum_children(self, index):
        children = self._child(index)
        return self.prob[children[0]]+self.prob[children[1]]

    def _update(self, index):
        self.prob[index] = self._sum_children(index)
        if index is not 0:
            self._update(self._parent(index))

    def _sample(self, value, index):
        if index >= self.start:
            return index
        children = self._child(index)
        if self.prob[children[0]] >= value:
            return self._sample(value, children[0])
        else:
            return self._sample(value-self.prob[children[0]],children[1])
