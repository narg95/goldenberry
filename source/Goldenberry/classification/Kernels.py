import numpy as np
from Goldenberry.classification.base.GbBaseKernel import BaseKernel

class LinealKernel(BaseKernel):
    """Linear kernel"""

    def compute(self, x, y):
        return x.dot(y)

class PolynomialKernel(BaseKernel):
    """Polynomial kernel"""

    def setup(self, degree):
        self.degree = degree

    def compute(self, x, y):
        return x.dot(y)**self.degree

class GaussianKernel(BaseKernel):
    """Gaussian kernel"""
    def setup(self, gamma):
        self.sqrgamma = gamma * gamma

    def compute(self, x, y):
        val = (x-y)
        return np.exp(-val.dot(val.T)/self.sqrgamma)