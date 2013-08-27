from Orange.core import Learner
import math
import numpy as np
import itertools as iter
import Orange

class Perceptron:
    """Perceptron algorithm"""

    class PerceptronResult:
        W, B, K, R
        
        def __init__(self):
            W = None
            B = 0.0
            K = 0
            R = 0.0

    def learn(self, (X, Y), p = PerceptronResult(), lr = 1.0):
        p.W = np.zeros(X.shape[1]) if None == p.W else p.W
        # max norm from training set.
        p.R = np.multiply(X, X).sum(axis=1).max()

        for xi, yi in iter.imap(lambda x,y : (x, -1 if y == 0 else y) ,X, Y):
            if yi*(p.W.dot(xi) + p.B) <= 0 :
                p.W += lr*yi*xi
                p.B += lr*yi*p.R
                p.K += 1
        return p

    def predict(self, X, p):
        return (X.dot(W.T) + B)/R        
                    
class PerceptronLearner(Learner):
    """Kernel perceptron learner"""
    
    def __init__(self, max_iter = 10, lr = 1.0, name = "Perceptron"):
        self.max_iter = max_iter
        self.lr = lr
        self.name = name
        
    def __call__(self,data,weight=0):
        """Learn from the given table of data instances."""
        
        examples = Orange.core.Preprocessor_dropMissingClasses(data)
        class_var = examples.domain.class_var

        if data.domain.class_vars:
            raise ValueError("multi-target classification is not supported.  Please provide a dataset with only class variable")

        if class_var == Orange.feature.Continuous:
                    raise ValueError("non-discrete classes not supported")
               
        if len(examples) == 0:
            raise ValueError("Example table is without any defined classes")

        n_classes = len(data.domain.class_var.values)
        y = np.array([int(d.get_class()) for d in data])
        n_classes = len(data.domain.class_var.values)
        if n_classes > 2:
            Y = np.eye(n_classes)[y]
        else:
            Y = y[:,np.newaxis]
        X, Y, _ = data.to_numpy()
        self.iters = 0
        perceptron = Perceptron()
        W, B = None, 0
        for i in range(self.max_iter):
            self.iters += 1 
            W, B, K = perceptron.learn((X,Y), (W,B), lr = self.lr)
            if K  == 0:
                break

        classifier = PerceptronClassifier(predict = perceptron.predict, W = W, B = B, domain = data.domain)
        return classifier
        
class PerceptronClassifier:
    
    def __init__(self,**kwargs):
        self.__dict__.update(**kwargs)

    def __call__(self,example, result_type = Orange.core.GetValue):
        input = np.array([[example[feature.name].value for feature in example.domain.features]])
        results = self.predict(input,(self.W, self.B))

        mt_value =  self.domain.class_var[0 if results[0] <= 0 else 1]
        frecuencies = [1.0, 0.0] if results[0] < 0 else [0.0, 1.0] if results[0] > 0 else [0.5, 0.5]
        mt_prob = Orange.statistics.distribution.Discrete(frecuencies)
        mt_prob.normalize()

        if result_type == Orange.core.GetValue: return tuple(mt_value) if self.domain.class_vars else mt_value
        elif result_type == Orange.core.GetProbabilities: return tuple(mt_prob) if self.domain.class_vars else mt_prob
        else: 
            return [tuple(mt_value), tuple(mt_prob)] if self.domain.class_vars else [mt_value, mt_prob] 
        return