from Goldenberry.optimization.base.GbCostFunction import GbCostFunction
from Goldenberry.classification.Kernels import WeightedKernel
from Orange.evaluation.testing import cross_validation
from Orange.evaluation.scoring import CA
import numpy as np

class WKieraCostFunction(GbCostFunction):
    """WKiera cost function."""

    def __init__(self, kernel, data,  learner_type, **learner_args):
        self.kernel = kernel
        self.learner_type = learner_type
        self.learner_args = learner_args
        self.data = data

    def execute(self, solutions):
        #TODO Omptimize for parallel execution
        learners = [None]*len(solutions)
        for idx, weight in enumerate(solutions):
            weigted_kernel = WeightedKernel(weight, self.kernel)
            learner = self.learner_type(kernel = weigted_kernel, **self.learner_args)
            learners[idx] = learner

        results = np.array(CA(cross_validation(learners, self.data, folds = 10)))
        self._update_statistics(results)
        return results

    def _update_statistics(self, results):
        for result in results:
            super(WKieraCostFunction, self)._update_statistics(result)