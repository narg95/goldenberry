from Goldenberry.optimization.cost_functions.functions import *
import numpy as np
from unittest import *

class CostFunctionTest(TestCase):
    
    def test_onemax(self):
        sample = np.array([0,1,0,1, 0])
        sample.shape = (1, 5)
        func = Onemax()
        self.assertEqual(func(sample), 2)

    def test_zero(self):
        sample = np.matrix([0,1,0,1,0])
        sample.shape = (1, 5)
        func = Zeromax()
        self.assertEqual(func(sample), 3)

    def test_condonemax_base(self):
        length = 10
        func = CondOnemax(length)
        self.assertTrue(np.equal(func.cond_indexes, range(length)).all())

    def test_condonemax_base_1(self):
        func = CondOnemax(cond_indexes =[9,8,7,6,5,4,3,2,1,0])
        self.assertTrue(np.equal(func.cond_indexes, [9,8,7,6,5,4,3,2,1,0]).all())

    def test_condonemax_cost_calculation(self):
        sample = np.array([0,1,1,1,1,0,1,0,1,1])
        sample.shape = (1, 10)
        func = CondOnemax(cond_indexes = [9,8,7,6,5,4,3,2,1,0])
        self.assertEqual(func.cost(sample)[0], 5.0)
        

if __name__ == '__main__':
    unittest.main()
