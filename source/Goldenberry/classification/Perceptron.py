from Orange.core import Learner
import math
import numpy as np
import itertools as iter
import Orange

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
                    
class PerceptronLearner(Learner):
    """Kernel perceptron learner"""
    
    # TODO:  Only remove this code when you make sure it is not required
    # for being integrated with Orange test learners.
    #@classmethod
    #def __new__( cls, *args, **kwargs):
    #    self = Orange.classification.Learner.__new__(cls, **kwargs)

    #    if dataset is None:   
    #        return self
    #    else:
    #        self.__init__(**kwargs)
    #        return self(dataset, weight)

    def __init__(self, max_iter = 1000, lr = 1.0):
        self.max_iter = max_iter
        self.lr = lr
        
    def __call__(self,data,weight=0):
        """Learn from the given table of data instances."""

        X, Y, _ = data.to_numpy()
        perceptron = Perceptron()
        W, B = None, 0
        for i in range(self.max_iter):
            W, B, K = perceptron.learn((X,Y), (W,B), lr = self.lr)
            if K  == 0:
                break

        classifier = PerceptronClassifier(predict = perceptron.predict, W = W, B = B)
        return classifier
        
class PerceptronClassifier:
    
    def __init__(self,**kwargs):
        self.__dict__.update(**kwargs)

    def __call__(self,example, result_type=Orange.core.GetValue):
        input = np.array([[example[feature.name].value for feature in example.domain.features]])
        result = self.predict(input,(self.W, self.B))
        return example.domain.class_var[0 if result[0] <= 0 else 1]