import numpy as np

def LinealKernel(x, y):
    return x.dot(y.T)

def PolynomialKernel(x, y):
    degree = 2
    return x.dot(y.T)**degree

def GaussianKernel(x, y):
    gamma = 1.0
    val = (x-y)
    return np.exp(-val.dot(val.T)/gamma)

class WeightedKernel():
    def __init__(self, weight, kernel):
        self.weight = weight
        self.kernel = kernel

    def __call__(self, x, y):
        return self.kernel(x*self.weight, y*self.weight)
