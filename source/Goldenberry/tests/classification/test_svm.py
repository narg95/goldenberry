from Goldenberry.classification.SvmLearner import *
from Goldenberry.classification.base.GbKernel import GbKernel
from Goldenberry.classification.Kernels import LinealKernel 
import orngSVM, Orange

from unittest import *

class SvmTest(TestCase):
    """Svm tests."""

    def test_basic(self):
        svm = SvmLearner()
        svm.kernel = lambda _: GbKernel(func = LinealKernel)
        self.assertTrue(svm.kernel is orngSVM.KernelWrapper)
        svm(data = Orange.data.Table("iris"))
