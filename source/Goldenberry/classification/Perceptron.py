from Orange.core import Learner
import numpy as np

class _Perceptron:
    """Perceptron algorithm"""

    _training_set = None
    _bias = 0
    _learning_rate = 1.0
    _R = 0
    _w = None

    def Learn(self, training_set, w = None, learning_rate = 1.0):
        self._w = w
        self._learning_rate = learning_rate
        _R = np.

        
class PerceptronLearner(Learner):
    """Kernel perceptron learner"""
    
    _learning_rate = 1.0
 
    @classmethod
    def __new__(cls, data = None, weight_id = 0, **argkw):
        self = Orange.classification.Learner.__new__(cls, **argkw)
        if data:
            self.__init__(**argkw)
            return self.__call__(data, weight_id)
        else:
            return self

    def __init__(self, learning_rate = 1.0):
        self._learning_rate = learning_rate
        
    def __call__(self,data,weight=0):
        """Learn from the given table of data instances."""
        

