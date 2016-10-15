from Goldenberry.classification.BudgetKernelPerceptron import *
from Goldenberry.classification.KernelPerceptron import *
from Goldenberry.classification.Kernels import *
from time import gmtime, strftime
import logging
import Orange
import os
import itertools as itert
import winsound
from unittest import *

class BudgetKernelPerceptronTest(TestCase):
    """Budget Kernel Perceptron Test"""

    def setUp(self):
        #Define Logger Configuration
        logger = logging.getLogger('budgetKernelPerceptron')
        if len(logger.handlers) == 0 :
            hdlr = logging.FileHandler('budgetKernelPerceptron.log')
            formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
            hdlr.setFormatter(formatter)
            logger.addHandler(hdlr) 
            logger.setLevel(logging.DEBUG)        
        self.logger = logger

    def test_budgetKernelPerceptron(self):
        fileName = "test_3classes_notseparable_data.tab"
        training_set = Orange.data.Table(os.path.dirname(__file__) + "\\" + fileName)
        # Take 1% of budget
        budget = training_set.__len__() * 0.01 
        max_iter = 1
        kernel = GaussianKernel()
        kernel.setup(0.1)
        fileKernel = "none"
        learner = BudgetKernelPerceptronLearner(max_iter = max_iter, budget = budget, kernel = kernel)
        classifier = learner(training_set)
        errors = 0
        success = 0
        self.logger.info("Classifying... ")
        for item in training_set:
            result = classifier(item)
            #self.assertEqual(result, item.getclass())
            if result == item.getclass():
                success += 1
            else:
                errors += 1
        self.logger.info("End classifying... " )
        msg = "BudgetKernelPerceptron ends with {0} errors and {1} successes using {2} learning iterations. File: {3} Learner {4} Budget {5} \n".format(errors, success, max_iter,fileName, learner.__class__.__name__, budget)
        self.logger.info(msg)
        