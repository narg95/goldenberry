from unittest import *
import Orange
from Goldenberry.feature_selection.WKiera import WKieraCostFunction
from Goldenberry.classification.Perceptron import PerceptronLearner
from Goldenberry.classification.base.GbKernel import GbKernel
from Goldenberry.classification.Kernels import LinealKernel
from Goldenberry.optimization.edas.Univariate import Pbil
import numpy as np

class WKieraTest(TestCase):
    
    def test_basic(self):
        data = Orange.data.Table("zoo")
        learner =  PerceptronLearner(max_iter = 1)
        learner.kernel = GbKernel(func = LinealKernel)
        cost_func = WKieraCostFunction(data, learner, solution_weight = 0.3)
        solutions = np.ones((2, len(data.domain.attributes)))
        cost = cost_func(solutions)
        self.assertTrue(((cost - np.array([0.61818182, 0.61818182])) < 0.001).all())
        



