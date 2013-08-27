import math
import numpy as np
import itertools as iter

class OneVsAllMulticlassLearner:
    """Multiclass learner algorithm."""
    
    def learn(self, (X,Y), binary_learner_type, max_iter = 10, n_classes = 2, *args):
        nY = np.eye(n_classes)*2 -1
        learners = [learner_type() for i in range(n_classes)]
        for i in range(i_classes):
            index = i
            learner = learner_type()
            result = None
            for j in range(self.max_iter):
                result = learner.learn((X,nY[i][Y]), result, *args)    
                if learner.has_learned(result):
                    break
            learners[index] = learner, result

    def predict(self, X, learners, *args):
        
                    
