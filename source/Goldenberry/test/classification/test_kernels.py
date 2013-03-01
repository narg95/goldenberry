import numpy as np
from unittest import *
from Goldenberry.classification.Kernels import *

class KernelsTest(TestCase):

    def test_linear_kernel(self):
        x = np.array([2,3,4])
        y = np.array([1,2,3])
        lin_ker = LinealKernel()
        self.assertEqual(lin_ker.compute(x, y), 20.0)

    def test_poly_kernel(self):
        x = np.array([2,3,4])
        y = np.array([1,2,3])
        pol_ker = PolynomialKernel()
        pol_ker.setup(3)
        self.assertEqual(pol_ker .compute(x, y), (20.0**3))