from unittest import *
import Orange
from Goldenberry.feature_selection.WKiera import GbWrapperCostFunction, test_solution
from Goldenberry.classification.Perceptron import PerceptronLearner
from Goldenberry.classification.base.GbKernel import GbKernel
from Goldenberry.classification.base.GbFactory import GbFactory
from Goldenberry.classification.Kernels import LinealKernel
from Goldenberry.optimization.edas.Univariate import Cga
from orngSVM import SVMLearner
import numpy as np
import time

class WKieraTest(TestCase):
    
    def test_basic(self):
        zoo_data = Orange.data.Table("zoo")
        indices2 = Orange.data.sample.SubsetIndices2(p0=0.5)
        ind = indices2(zoo_data)
        data = zoo_data.select(ind, 0)
        test_data = zoo_data.select(ind, 1)

        factory =  GbFactory(PerceptronLearner, {"max_iter" : 1, "kernel": LinealKernel})
        cost_func = GbWrapperCostFunction(data, test_data, factory, solution_weight = 0.1, test_folds = 5)
        solutions = np.ones((2, len(data.domain.attributes)))
        cost = cost_func(solutions)
        self.assertTrue(((cost - np.array([0.75845455,  0.75845455])) < 0.001).all())

    def test_test_solution(self):
        tic = time.time()
        zoo_data = Orange.data.Table("zoo")
        indices2 = Orange.data.sample.SubsetIndices2(p0=0.5)
        ind = indices2(zoo_data)
        data = zoo_data.select(ind, 0)
        test_data = zoo_data.select(ind, 1)
        learner =  GbFactory(SVMLearner, {"normalization":0})
        results = [None]
        cost_func = GbWrapperCostFunction(data, test_data, learner, test_folds = 5)
        cga = Cga()
        cga.setup(max_evals = 20, callback_func = self.progress)
        cga.cost_func = cost_func
        cga.search()
        toc = time.time() - tic
        print "%s"%toc

    def progress(self, result, progress):
        print "%s"%progress