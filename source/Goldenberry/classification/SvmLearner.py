from Orange.core import Learner
import orngSVM
from orngSVM import SVMLearner
import numpy as np

class SvmLearner(SVMLearner):
    """SVM Learner algorithm"""

    def __init__(self, C=1.0, nu=0.5, p=0.1, gamma=0.0, degree=3,
                 coef0=0, shrinking=True, probability=True, verbose=False,
                 cache_size=200, eps=0.001, normalization=True,
                 weight=[], **kwargs):
        super(SvmLearner, self).__init__(svm_type= orngSVM.SVMLearner.Nu_SVC, kernel_type = orngSVM.SVMLearner.Custom, 
                                         C=C, nu=nu, p=p, gamma=gamma, degree=degree, coef0=coef0, shrinking=shrinking, 
                                         probability=probability, verbose=verbose,cache_size=cache_size, eps=eps, 
                                         normalization=normalization,weight=weight,**kwargs)

    @property
    def kernel(self):
        return self.kernel_func

    @kernel.setter
    def kernel(self, value):
        self.kernel_func = self.wrap_kernel(value)

    def wrap_kernel(self, kernel_func):
        if None is kernel_func:
            return None

        kernel = kernel_func
        return orngSVM.KernelWrapper(lambda x,y: kernel(*to_numpy(x,y)))

def to_numpy(i1, i2):
    x = np.array([i for i in i1 if i.native() != i1.get_class().native()], dtype = float)
    y = np.array([i for i in i2 if i.native() != i2.get_class().native()], dtype = float)
    return x,y