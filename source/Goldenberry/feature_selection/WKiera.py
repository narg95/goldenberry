from Goldenberry.optimization.base.GbCostFunction import GbCostFunction
from Goldenberry.classification.Kernels import WeightedKernel
from Orange.evaluation.testing import cross_validation
from Orange.evaluation.scoring import CA
import numpy as np
from copy import deepcopy

class WKieraCostFunction(GbCostFunction):
    """WKiera cost function."""

    def __init__(self, kernel, data,  learner, solution_weight = 0.1):
        self.reset_statistics()
        self.kernel = kernel
        self.learner = learner
        self.data = data
        self.solution_weight = solution_weight
        self.var_size = len(data.domain.attributes)

    def execute(self, solutions):
        #TODO Omptimize for parallel execution
        learners = [None]*len(solutions)
        for idx, weight in enumerate(solutions):
            weighted_kernel = WeightedKernel(weight, self.kernel)
            learner = deepcopy(self.learner)
            learner.kernel = weighted_kernel
            learners[idx] = learner

        results = np.array(CA(cross_validation(learners, self.data, folds = 10)))
        self._update_statistics(results)
        return results*(1 - self.solution_weight) + (1 - np.average(solutions, axis = 1)) * self.solution_weight

    def _update_statistics(self, results):
        for result in results:
            super(WKieraCostFunction, self)._update_statistics(result)