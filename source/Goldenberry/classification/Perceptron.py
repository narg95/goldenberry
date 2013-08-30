from Orange.core import Learner
from Goldenberry.classification.MulticlassLearner import OneVsAllMulticlassLearner
import math
import numpy as np
import itertools as iter
import Orange

class Perceptron:
    """Perceptron algorithm"""

    def __init__(self, W = None, B= 0.0, K = None, lr = 1.0):
        self.W = W
        self.B = B
        self.K = K
        self.R = 0.0
        self.lr = lr
        self.iters = 0
        self.acc_K = 0
    
    def has_learned(self):
        return self.K == 0

    def learn(self, (X, Y)):
        # exits if no mistakes where found in the last run
        if self.has_learned():
            return
        
        self.K = 0
        self.W = np.zeros(X.shape[1]) if None == self.W else self.W
        
        # max norm from training set.
        self.R = np.multiply(X, X).sum(axis=1).max()

        for xi, yi in iter.izip(X, Y):
            if yi*(self.W.dot(xi) + self.B) <= 0 :
                self.W += self.lr*yi*xi
                self.B += self.lr*yi*self.R
                self.K += 1
        self.iters += 1
        self.acc_K += self.K

    def predict(self, X):
        score = (X.dot(self.W.T) + self.B)/self.R
        return np.sign(score), score
                    
class PerceptronLearner(Learner):
    """Kernel perceptron learner"""
    
    def __init__(self, max_iter = 10, lr = 1.0, name = "Perceptron", one_vs_all = True):
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

        X, Y, _ = data.to_numpy()
        n_classes = len(data.domain.class_var.values)
        if n_classes > 2:
            learner = OneVsAllMulticlassLearner(Perceptron, n_classes, lr = self.lr)
        else:
            Y = Y * 2 - 1
            learner = Perceptron(lr = self.lr)
        
        for j in range(self.max_iter):
            learner.learn((X, Y))
            if learner.has_learned():
                break            

        classifier = PerceptronClassifier(learner = learner, domain = data.domain)
        return classifier
        
class PerceptronClassifier:
    
    def __init__(self,**kwargs):
        self.__dict__.update(**kwargs)

    def __call__(self,example, result_type = Orange.core.GetValue):
        input = np.array([[example[feature.name].value for feature in example.domain.features]])
        # takes the index 0 because it is expected to have only one result per prediction, one by one classification.
        # it seems to be a restriction when integrating with the Orange Canvas.
        prediction, scores = self.learner.predict(input)
        
        mt_prob = []
        mt_value = []

        # multi-objective prediction
        if self.domain.class_vars:
            raise ValueError("multi-objective prediction is not supported.")
        
        n_classes = len(self.domain.class_var.values)
        if n_classes == 2:
            results = np.maximum([0.5 - scores[0], 0.5 + scores[0]], 0.0).tolist()
        else:
            min = np.abs(np.min(scores))
            results = (scores[0] + min).tolist()

        cprob = Orange.statistics.distribution.Discrete(results)
        cprob.normalize()

        mt_prob = cprob
        mt_value = Orange.data.Value(self.domain.class_var, cprob.values().index(max(cprob)))

        if result_type == Orange.core.GetValue: 
            return mt_value
        elif result_type == Orange.core.GetProbabilities: 
            return mt_prob
        else: 
            return [mt_value, mt_prob] 