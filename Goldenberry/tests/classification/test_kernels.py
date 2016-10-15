import numpy as np
from unittest import *
from Goldenberry.classification.Kernels import *
import orange, orngSVM

class KernelsTest(TestCase):

    def test_linear_kernel(self):
        x = np.array([2,3,4])
        y = np.array([1,2,3])
        self.assertEqual(LinealKernel(x, y), 20.0)

    def test_poly_kernel(self):
        x = np.array([2,3,4])
        y = np.array([1,2,3])
        self.assertEqual(PolynomialKernel(x, y), (20.0**2))
