from Goldenberry.classification.SvmLearner import *
from Goldenberry.classification.base.GbKernel import GbKernel
from Goldenberry.classification.Kernels import LinealKernel 
import orngSVM, Orange

from unittest import *

class SvmTest(TestCase):
    """Svm tests."""

    def test_basic(self):
        svm = SvmLearner()
        svm.kernel = GbKernel(func = LinealKernel)
        self.assertTrue(type(svm.kernel) is orngSVM.KernelFunc)
        svm(data = Orange.data.Table("iris"))
