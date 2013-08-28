import math
import numpy as np
import itertools as iter

class OneVsAllMulticlassLearner:
    """Multiclass learner algorithm."""
    
    def __init__(self, learner_type, n_classes, max_iter=10, **kargs):
        self.nY = np.eye(n_classes,dtype= int) * 2 - 1
        self.learners = [learner_type(**kargs) for i in range(n_classes)]
        self.n_classes = n_classes
        self.max_iter = max_iter
    
    def learn(self, (X,Y)):
        for i in range(self.n_classes):
            index = i
            learner = self.learners[i]
            for j in range(self.max_iter):
                learner.learn((X,self.nY[i][Y.astype(int)]))    
                if learner.has_learned():
                    break            

    def predict(self, X):
        classification = np.array([np.maximum(classifier.predict(X), 0.0) for classifier in self.learners])
        return np.argmax(classification ,axis = 0)