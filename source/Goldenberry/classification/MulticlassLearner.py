import math
import numpy as np
import itertools as iter

class OneVsAllMulticlassLearner:
    """Multiclass learner algorithm."""
    iters = 0

    def has_learned(self):
        return np.all([learner.has_learned() for learner in self.learners])

    def __init__(self, learner_type, n_classes, **kargs):
        self.nY = np.eye(n_classes,dtype= int) * 2 - 1
        self.learners = [learner_type(**kargs) for i in range(n_classes)]
        self.n_classes = n_classes
    
    def learn(self, (X,Y)):
        int_Y = Y.astype(int)
        for i in range(self.n_classes):
            index = i
            learner = self.learners[i]
            learner.learn((X, mask(self.nY, int_Y, i)))
        self.iters += 1

    def predict(self, X):
        classification = np.array([np.maximum(classifier.predict(X)[1], 0.0) for classifier in self.learners])
        class_index = np.argmax(classification ,axis = 0)
        return class_index, classification.T

def mask(eye, int_Y, class_index):
    return eye[class_index][int_Y]