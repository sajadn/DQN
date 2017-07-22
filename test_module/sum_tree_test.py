from ..algorithms.memory_strategy.sum_tree import SumTree
import unittest

class sumTreeTest(unittest.TestCase):
    def setUp(self):
        self.data = [1,4,5,6,7,3,20]
        prob = [1,4,5,7,8,9,10]
        self.st = SumTree(7, self.data, prob)

    def test_check_initial_values(self):
        result = [44,17, 27, 5, 12, 17, 10, 1, 4, 5, 7, 8, 9, 10, 0]
        
        self.assertListEqual(self.st.prob.tolist(), result)
        temp =  self.data + [0]
        self.assertListEqual(self.st.data.tolist(), temp)

    def test_check_insert(self):
        result = [50,17, 33, 5, 12, 17, 16, 1, 4, 5, 7, 8, 9, 10, 6]
        self.st.insert(3,6)
        self.assertListEqual(self.st.data.tolist(), self.data + [3])
        self.assertListEqual(self.st.prob.tolist(), result)
        self.st.insert(7,8)
        result = [57,24, 33, 12, 12, 17, 16, 8, 4, 5, 7, 8, 9, 10, 6]
        self.assertListEqual(self.st.data.tolist(), [7,4,5,6,7,3,20,3])
        self.assertListEqual(self.st.prob.tolist(), result)

    def test_check_update(self):
        result = [39,12, 27, 5, 7, 17, 10, 1, 4, 0, 7, 8, 9, 10, 0]
        self.st.update(9, 0)
        self.assertListEqual(self.st.data.tolist(), self.data +[0])
        self.assertListEqual(self.st.prob.tolist(), result)

    def test_check_sample(self):
        v, p, i = self.st.sample(5)
        self.assertEqual(len(v), 5)
        self.assertEqual(len(i),len(set(i)))
