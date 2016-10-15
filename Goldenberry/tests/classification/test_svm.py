from Goldenberry.classification.SvmLearner import *
from Goldenberry.classification.base.GbKernel import GbKernel
from Goldenberry.classification.Kernels import LinealKernel 
import orngSVM, Orange, orange
import time

from unittest import *

class SvmTest(TestCase):
    """Svm tests."""

    def test_basic(self):
        svm = SvmLearner()
        svm.kernel = GbKernel(func = LinealKernel)
        self.assertTrue(type(svm.kernel) is orngSVM.KernelFunc)
        svm(data = Orange.data.Table("iris"))

    def test_basic_performance(self):
        data = orange.ExampleTable("iris.tab")
        
        tic = time.time()
        data = orange.ExampleTable("iris.tab")
        l1 = orngSVM.SVMLearner()
        l1.kernel_func = orngSVM.KernelWrapper(my_kernel)
        l1.kernel_type =orange.SVMLearner.Custom
        l1.probability = False
        c1 = l1(data)
        toc = time.time() - tic
        print "With Wrapper: %s"%toc

        tic = time.time()
        l1 = SvmLearner()
        l1.kernel = GbKernel(func = LinealKernel)
        l1.probability = False
        c1 = l1(data)
        toc = time.time() - tic
        print "Lambda Wrapper: %s"%toc

        tic = time.time()
        l1 = orngSVM.SVMLearner()
        l1.kernel_type = orange.SVMLearner.Linear
        l1.probability = False
        c1 = l1(data)
        toc = time.time() - tic
        print "Native Wrapper: %s"%toc


    def test_integration_SVM(self):
        data = orange.ExampleTable("iris.tab")
        l1 = orngSVM.SVMLearner()
        l1.kernel_func = orngSVM.KernelWrapper(MyKernel().my_kernel)
        l1.kernel_type =orange.SVMLearner.Custom
        l1.probability = True
        c1 = l1(data)    

class MyKernel:
    
    def __call__(self, i1, i2):
        return self.my_kernel(i1, i2)

    def my_kernel(self, i1, i2):
        x = np.array([i for i in i1 if i.native() != i1.get_class().native()], dtype = float)
        y = np.array([i for i in i2 if i.native() != i2.get_class().native()], dtype = float)
        return x.dot(y)

def my_kernel(i1, i2):
        classx = i1.get_class().value
        classy = i2.get_class().value
        x = np.array([i.value for i in i1 if i.value != classx])
        y = np.array([i.value for i in i2 if i.value != classy])
        return x.dot(y)
