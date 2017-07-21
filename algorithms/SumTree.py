import math
import numpy as np
import random

class SumTree:
    _sum_leafs = 0
    _cursor = 0
    def __init__(self, size):
        self.tree_levels = math.ceil(math.log(size, 2))+1
        self.node_counts = (2**tree_levels)-1
        self.data = np.zeros(node_counts)

    def initial_values(self, values):
        start = 2**(self.tree_levels-1)-1
        for index, value in enumerate(values):
            data[start + index] = value
            _sum_leafs += value
        cursor = index
        for level in range(self.tree_levels - 1, 0, -1):
            for i in range(2**(level-1)-1, 2**(level)-1, 1)
                data[i] = sum(self._child(i))

    def add(self, index, value):
        data[index] = value
        _update(index, value)


    def _parent(self, index):
        return i/2 if(i%2==0) else (i-1)/2

    def _child(self, index):
        return index*2+1, index*2+2

    def _update(self, index):
        data[index] = sum(_child(index))
        if index is not 0:
            _update(_parent(index))

    def update(self, index, value):
        data[index] = value

    def _sample(self, value, index):
        if index >= 2**self.tree_levels-1
            return data[index]
        childs = self._child(index)
        if data[childs[0]] >= value:
            return _sample(value, childs[0])
        else:
            return _sample(value-data[index],childs[1])

    def sample(self, size):
        random_numbers = random.sample(range(_sum_leafs), size)
        [_sample(rn, 0) for rn in random_numbers]
