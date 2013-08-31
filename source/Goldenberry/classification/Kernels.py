import numpy as np
from Goldenberry.classification.base.GbBaseKernel import BaseKernel

def LinealKernel(x, y):
    return x.dot(y)

def PolynomialKernel (x, y):
    degree = 3    
    return x.dot(y)**degree

def GaussianKernel(x, y):
    gamma = 1.0
    val = (x-y)
    return np.exp(-val.dot(val.T)/gamma)