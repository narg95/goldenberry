from Goldenberry.optimization.base.GbCostFunction import GbCostFunction
from Orange.evaluation.testing import cross_validation
from Orange.evaluation.scoring import CA
import numpy as np
from copy import deepcopy
import Orange
import threading as th
import Queue

class WKieraCostFunction(GbCostFunction):
    """WKiera cost function."""

    def __init__(self, data,  learner, solution_weight = 0.1, folds = 10):
        self.reset_statistics()
        self.learner = learner
        self.data = data
        self.folds = folds
        self.solution_weight = solution_weight
        self.var_size = len(data.domain.attributes)

    def execute(self, solutions):
        results = np.empty(len(solutions), dtype = float)
        threads = [None] * len(solutions)
        for idx, weight in enumerate(solutions):                    
            thread = th.Thread(target = test_solution, args = [self.learner, weight, self.data, results, idx, self.folds])
            thread.start()
            threads[idx] = thread
        
        #sync all threads
        for thread in threads:
            thread.join()

        self._update_statistics(results)
        return results*(1 - self.solution_weight) + (1 - np.average(solutions, axis = 1)) * self.solution_weight

    def _update_statistics(self, results):
        for result in results:
            super(WKieraCostFunction, self)._update_statistics(result)

def test_solution(learner, weight, data, results, idx, folds):
    #transforms the data base on the weights given by each solution.    
    weighted_data = data.to_numpy("ac")[0] * np.concatenate((weight, [1]))
    results[idx] = CA(cross_validation([deepcopy(learner)], Orange.data.Table(data.domain, weighted_data), folds = folds))[0]