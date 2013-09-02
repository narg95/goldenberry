import numpy as np
from unittest import *
from Goldenberry.classification.Kernels import *

class KernelsTest(TestCase):

    def test_linear_kernel(self):
        x = np.array([2,3,4])
        y = np.array([1,2,3])
        self.assertEqual(LinealKernel(None, x, y), 20.0)

    def test_poly_kernel(self):
        x = np.array([2,3,4])
        y = np.array([1,2,3])
        self.assertEqual(PolynomialKernel(None, x, y), (20.0**3))