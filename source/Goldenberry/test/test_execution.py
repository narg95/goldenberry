import unittest
import os
import sys
import inspect
import imp

if __name__ == "__main__":

    test_modules = [
                    'optimization.cost_functions.tests.CostFunctionTest',
                    'optimization.edas.test_edas.CgaTest',
                    'optimization.edas.test_edas.BmdaTest',
                    'statistics.test_distributions.BinomialTest',
                    'statistics.test_distributions.BivariateBinomialTest',
                    'statistics.test_distributions.BinomialContingencyTableTest',
                    'classification.test_perceptron.PerceptronTest',
                    'classification.test_kernels.KernelsTest']    
    suite = unittest.TestLoader().loadTestsFromNames(test_modules)    
    runner = unittest.TextTestRunner(verbosity = 2)
    runner.run(suite)

    

    