from unittest import *
import Orange
from Goldenberry.feature_selection.WKiera import WKieraCostFunction
from Goldenberry.classification.Perceptron import PerceptronLearner
from Goldenberry.classification.base.GbKernel import GbKernel
from Goldenberry.classification.Kernels import LinealKernel
from Goldenberry.optimization.edas.Univariate import Pbil

class WKieraTest(TestCase):
    
    def test_basic(self):
        data = Orange.data.Table("zoo")
        cost_func = WKieraCostFunction(GbKernel(func = LinealKernel), data, PerceptronLearner(max_iter = 1))
        opt = Pbil()
        opt.cost_func = cost_func
        opt.setup(cand_size = 10, learning_rate = 0.8)
        #best_weights = opt.search()



