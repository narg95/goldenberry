import numpy as np

def LinealKernel(self, x, y):
    return x.dot(y)

def PolynomialKernel(self, x, y):
    degree = 3    
    return x.dot(y)**degree

def GaussianKernel(self, x, y):
    gamma = 1.0
    val = (x-y)
    return np.exp(-val.dot(val.T)/gamma)