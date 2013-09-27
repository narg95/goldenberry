import unittest
import os
import sys
import inspect
import imp
import optimization.cost_functions
from msvcrt import getch


if __name__ == "__main__":

    test_modules = [
                    'feature_selection.test.WKieraTest' ,
                    'optimization.cost_functions.CostFunctionTest' ,
                    'optimization.edas.test_edas.CgaTest',
                    'optimization.edas.test_edas.PbilTest',
                    'optimization.edas.test_edas.BmdaTest',
                    'optimization.edas.test_edas.TildaTest',
                    'optimization.edas.test_edas.PbilcTest',
                    'optimization.edas.test_edas.OptmizerTesterTest',
                    'statistics.test_distributions.BinomialTest',
                    'statistics.test_distributions.BivariateBinomialTest',
                    'statistics.test_distributions.BinomialContingencyTableTest',
                    'statistics.test_distributions.GaussianTest',
                    'statistics.test_distributions.GaussianTruncTest',
                    'classification.test_perceptron.PerceptronTest',
                    'classification.test_multiclass_learner.MulticlassLearnerTest',
                    'classification.test_kernels.KernelsTest'
                    ]    
    suite = unittest.TestLoader().loadTestsFromNames(test_modules)    
    runner = unittest.TextTestRunner(verbosity = 2)
    runner.run(suite)
    

    

    