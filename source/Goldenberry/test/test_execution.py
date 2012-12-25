import unittest
import os
import sys
import inspect
import imp

if __name__ == "__main__":

    test_modules = [
                    'optimization.edas.test_cga.CgaTest',
                    'optimization.cost_functions.tests.CostFunctionTest',
                    'statistics.test_distributions.BinomialTest',
                    'statistics.test_distributions.BivariateBinomialTest',
                    'statistics.test_distributions.BinomialContingencyTableTest',
                    'classification.test_perceptron.PerceptronTest']    
    suite = unittest.TestLoader().loadTestsFromNames(test_modules)    
    runner = unittest.TextTestRunner(verbosity = 2)
    runner.run(suite)

    

    