from Goldenberry.classification.KernelPerceptron import *
from Goldenberry.classification.Kernels import *
from time import gmtime, strftime
import logging
import Orange
import os
import itertools as itert

from unittest import *

class KernelPerceptronTest(TestCase):
    """Kernel Perceptron Test"""

    def setUp(self):
        #Define Logger Configuration
        logger = logging.getLogger('kernelPerceptron')
        hdlr = logging.FileHandler('kernelPerceptron.log')
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr) 
        logger.setLevel(logging.DEBUG)        
        self.logger = logger

        
    def test_kernelPerceptron_two_classes(self):
        fileName = "test_data_2d.tab"
        training_set = Orange.data.Table(os.path.dirname(__file__) + "\\" + fileName)
        X, Y, _ = training_set.to_numpy()
        kernel = LinealKernel()
        kernelperceptron = KernelPerceptron(kernel, None)
        K = None
        sv_x, sv_y, sv_alpha = None, None, None
        
        #adds a column of ones to the dataset (bias)
        Xdata = np.hstack((np.ones((X.shape[0], 1)), X))
        Yclass = np.zeros(Y.shape)   
        # Set the Y class to -1, instead of 0.     
        for i in range(len(Yclass)):
            Yclass[i] = -1 if Y[i] == 0 else Y[i]

        iterations = 0
        while None == K or K > 0 and iterations < 15:
            sv_x, sv_y, sv_alpha, K = kernelperceptron.learn((Xdata, Yclass),(sv_x, sv_y, sv_alpha))
            iterations += 1
        
        prediction = kernelperceptron.predict(Xdata, (sv_x, sv_y * sv_alpha))
        for yp, yi in zip(prediction, Yclass):
            self.assertEqual(yp[0], yi)

        print("\n KernelPerceptron ends after {0} iterations ".format(iterations))

    def test_perceptor_learner(self):
        fileName = "test_3classes_separable_data.tab"
        training_set = Orange.data.Table(os.path.dirname(__file__) + "\\" + fileName)
        max_iter = 1
        learner = KernelPerceptronLearner(max_iter = max_iter)
        classifier = learner(training_set)
        self.assertGreaterEqual(max_iter, learner.iters)
        self.assertIsNotNone(classifier.predict)
        self.assertIsNotNone(classifier.domain)

    def test_kernelPerceptron_three_classes_separable_linear(self):
        max_iter = 4
        fileName = "test_3classes_separable_data.tab"
        training_set = Orange.data.Table(os.path.dirname(__file__) + "\\" + fileName)
        
        self.logger.info("Creating Learner... " )
        learner = KernelPerceptronLearner(max_iter = max_iter)
        self.logger.info("Learning... " )
        classifier = learner(training_set)
        errors = 0
        success = 0
        self.logger.info("Classifying... ")
        for item in training_set:
            result = classifier(item)
            self.assertEqual(result, item.getclass())
            
    def test_kernelPerceptron_three_classes_notseparable_gaussian(self):
        max_iter = 4
        fileName = "test_3classes_notseparable_data.tab"
        training_set = Orange.data.Table(os.path.dirname(__file__) + "\\" + fileName)
        
        self.logger.info("Creating Learner... " )
        kernel = GaussianKernel()
        kernel.setup(0.1)
        learner = KernelPerceptronLearner(max_iter = max_iter, kernel = kernel)
        self.logger.info("Learning... " )
        classifier = learner(training_set)
        errors = 0
        success = 0
        self.logger.info("Classifying... ")
        for item in training_set:
            result = classifier(item)
            self.assertEqual(result, item.getclass())
