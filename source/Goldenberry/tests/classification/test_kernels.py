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

    def test_integration_SVM(self):
        data = orange.ExampleTable("iris.tab")
        l1 = orngSVM.SVMLearner()
        l1.kernel_func = orngSVM.KernelWrapper(my_kernel)
        l1.kernel_type =orange.SVMLearner.Custom
        l1.probability = True
        c1 = l1(data)    

def my_kernel(i1, i2):
        x = np.array([i for i in i1 if i.native() != i1.get_class().native()], dtype = float)
        y = np.array([i for i in i2 if i.native() != i2.get_class().native()], dtype = float)
        return x.dot(y)
    
