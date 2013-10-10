from Goldenberry.optimization.base.GbCostFunction import GbCostFunction
from Orange.evaluation.testing import cross_validation
from Orange.evaluation.scoring import CA
import numpy as np
from copy import deepcopy
import Orange

class WKieraCostFunction(GbCostFunction):
    """WKiera cost function."""

    def __init__(self, data,  learner, solution_weight = 0.1):
        self.reset_statistics()
        self.learner = learner
        self.data = data
        self.solution_weight = solution_weight
        self.var_size = len(data.domain.attributes)

    def execute(self, solutions):
        #TODO Omptimize for parallel execution
        results = np.empty(len(solutions))
        for idx, weight in enumerate(solutions):
            #transforms the data base on the weigths given by each solution.
            weighted_data = self.data.to_numpy("ac")[0] * np.concatenate((weight, [1]))
            results[idx] = CA(cross_validation([deepcopy(self.learner)], Orange.data.Table(self.data.domain, weighted_data), folds = 10))[0]
        
        self._update_statistics(results)
        return results*(1 - self.solution_weight) + (1 - np.average(solutions, axis = 1)) * self.solution_weight

    def _update_statistics(self, results):
        for result in results:
            super(WKieraCostFunction, self)._update_statistics(result)

    def weight_data(self, data, solution):
        new_data = data.to_numpy("ac") * np.concatenate((solution,[1]))