from Orange.core import Learner
import math
import numpy as np
import itertools as iter

class Perceptron:
    """Perceptron algorithm"""

    def learn(self, (X, Y), (W, B) = (None, 0), lr = 1.0):
        W = np.zeros(X.shape[1]) if None == W else W
        K = 0
        # max norm from training set.
        R = np.multiply(X, X).sum(axis=1).max()

        for xi, yi in iter.imap(lambda x,y : (x, -1 if y == 0 else y) ,X, Y):
            if yi*(W.dot(xi) + B) <= 0 :
                W += lr*yi*xi
                B += lr*yi*R
                K += 1
        return W, B, K

    def predict(self, X, (W, B)):
        return np.sign(X.dot(W.T) + B)
                    
#class PerceptronLearner(Learner):
#    """Kernel perceptron learner"""
    
#    _learning_rate = 1.0
 
#    @classmethod
#    def __new__(cls, data = None, weight_id = 0, **argkw):
#        self = Orange.classification.Learner.__new__(cls, **argkw)
#        if data:
#            self.__init__(**argkw)
#            return self.__call__(data, weight_id)
#        else:
#            return self

#    def __init__(self, learning_rate = 1.0):
#        self._learning_rate = learning_rate
        
#    def __call__(self,data,weight=0):
#        """Learn from the given table of data instances."""
        

