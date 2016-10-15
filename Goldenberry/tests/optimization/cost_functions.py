from Goldenberry.optimization.cost_functions import *
from Goldenberry.optimization.base.GbCostFunction import GbCostFunction
import numpy as np
from unittest import *

class CostFunctionTest(TestCase):

    def test_onemax(self):
        sample = np.array([[0,1,0,1, 0]])
        func = GbCostFunction(OneMax)
        self.assertEqual(func(sample), 2)

    def test_statistics(self):
        sample1, cost1 = np.array([[0,1,1,1,0]]), 3.0
        sample2, cost2 = np.array([[0,0,0,1,1]]), 2.0
        sample3, cost3 = np.array([[0,1,1,1,1]]), 4.0
        func = GbCostFunction(OneMax)
        exp_cost1 = func(sample1)
        exp_cost2 = func(sample2)
        exp_cost3 = func(sample3)
        evals, argmin, argmax, min, max, mean, stdev = func.statistics()
        self.assertEqual((evals, argmin, argmax, min, max, mean), (3,2,3,cost2,cost3,3.0))
        self.assertAlmostEqual(stdev, np.sqrt(2.0/3.0), 5)


    def test_zero(self):
        sample = np.array([[0,1,0,1,0]])
        func = GbCostFunction(ZeroMax)
        self.assertEqual(func(sample), 3)

    def test_LeadingOnesBlocks(self):
        sample = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1])
        self.assertEqual(LeadingOnesBlocks(sample), 1)
        sample = np.array([1, 1, 1, 1, 0, 1, 1, 0, 1])
        self.assertEqual(LeadingOnesBlocks(sample), 1)
        sample = np.array([1, 1, 1, 1, 1, 1, 0, 1, 0])
        self.assertEqual(LeadingOnesBlocks(sample), 2)
        sample = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.assertEqual(LeadingOnesBlocks(sample), 3)
    
    #def test_condonemax_base(self):
    #    length = 10
    #    func = CondOnemax()
    #    sample = np.array([[0,1,1,1,1,0,1,0,1,1]])
    #    self.assertEqual(func.cost(sample), 4.0)

    #def test_condonemax_base_1(self):
    #    func = CondOnemax(cond_indexes =[9,8,7,6,5,4,3,2,1,0])
    #    self.assertTrue(np.equal(func.cond_indexes, [9,8,7,6,5,4,3,2,1,0]).all())

    #def test_condonemax_cost_calculation(self):
    #    sample = np.array([[0,1,1,1,1,0,1,0,1,1]])
    #    func = CondOnemax(cond_indexes = [9,8,7,6,5,4,3,2,1,0])
    #    self.assertEqual(func.cost(sample)[0], 5.0)
    
    def test_custom_function(self):
        sample = np.array([[0,1,0,1,0]])
        cust = GbCostFunction(script = "def mycustomfunction(solution):\n\treturn solution.sum() - 1.0")
        self.assertEqual(cust.cost(sample), 1.0)

if __name__ == '__main__':
    unittest.main()
