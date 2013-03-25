import unittest
import os
import sys
import inspect
import imp
import optimization.cost_functions
from optimization.cost_functions import CostFunctionTest

if __name__ == "__main__":

    test_modules = [
                    'optimization.cost_functions.CostFunctionTest',
                    'optimization.edas.test_edas.CgaTest',
                    'optimization.edas.test_edas.PbilTest',
                    'optimization.edas.test_edas.BmdaTest',
                    'optimization.edas.test_edas.TildaTest',
                    'statistics.test_distributions.BinomialTest',
                    'statistics.test_distributions.BivariateBinomialTest',
                    'statistics.test_distributions.BinomialContingencyTableTest',
                    'classification.test_perceptron.PerceptronTest',
                    'classification.test_kernels.KernelsTest'
                    ]    
    suite = unittest.TestLoader().loadTestsFromNames(test_modules)    
    runner = unittest.TextTestRunner(verbosity = 2)
    runner.run(suite)

    

    