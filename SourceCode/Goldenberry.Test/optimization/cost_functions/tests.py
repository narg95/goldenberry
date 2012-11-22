from Goldenberry.optimization.cost_functions.functions import *
import numpy as np
from unittest import *

class CostFunctionTest(TestCase):
    
    def test_onemax(self):
        sample = np.matrix([0,1,0,1, 0])
        func = onemax()
        self.assertEqual(func(sample), 2)

    def test_zero(self):
        sample = np.matrix([0,1,0,1, 0])
        func = zero()
        self.assertEqual(func(sample), -2)

if __name__ == '__main__':
    unittest.main()
