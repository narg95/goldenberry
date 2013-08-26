from Orange.core import Learner
from Goldenberry.classification.Kernels import *
from time import gmtime, strftime
import Orange
import math
import numpy as np
import itertools as iter
import Orange
import logging
import os

class KernelPerceptron:
    """Kernel Perceptron Algorithm"""

    def __init__(self, kernel, GRAM):
        self.kernel = kernel
        self.sv_alpha = None
        self.GRAM = GRAM

    # We assume that the classes for  y are 1 and -1.
    def learn(self, (X, Y), (sv_x, sv_y, sv_alpha) = (None, None, None), budget = 100):
        n_samples, n_features = X.shape
        # errors
        K = 0

        #Alpha, the size is the amount of samples
        if self.sv_alpha == None:
            self.sv_alpha = np.zeros(n_samples) 
        
        # If Gram matrix is not calculated, then calculated.
        # Watch out to free up the resources. Set the variables to None
        # in case of Memory Errors
        if self.GRAM == None:
            self.GRAM = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(n_samples):
                    self.GRAM[i,j] = self.kernel.compute(X[i], X[j])
        
        # Learn each sample. Increase the alpha in case the sample is missclasified.
        # Keep count of the errors (K) 
        for i in range(n_samples):
            if np.sign(np.sum(self.GRAM[:,i] * self.sv_alpha * Y)) != Y[i]:
                self.sv_alpha[i] += 1.0
                K += 1
        
        # Support vectors: Boolean Vector 
        # Flags for the Support Vectors that has an alpha greater than zero
        sv_flags = self.sv_alpha > 0
        #np.arange creates an array strarting from index 0 to the value passed by parameter  - 1. These are the indexes
        indexes = np.arange(n_samples)[sv_flags]        
        
        # return the data, weights and classes of the samples that make mistakes
        sv_x = X[indexes]
        sv_y = Y[indexes]
        sv_alpha = self.sv_alpha[indexes]
        return sv_x, sv_y, sv_alpha, K
    
    # Predict the class
    # sv_alpha: already multiply the class with the weight
    def predict(self, X, (sv_x, sv_alpha)):
        n_samples = X.shape[0]
        if (X.shape[1] + 1) == sv_x.shape[1] :
            X =  np.hstack( (np.ones( (n_samples, 1)), X))

        predictions = np.zeros((n_samples,2))
        for i in range(n_samples):        
            sum = 0        
            for aj, xj in zip(sv_alpha, sv_x):
                sum += aj * self.kernel.compute(xj, X[i])
            predictions[i] = (np.sign(sum), sum)
        return predictions

class KernelPerceptronLearner(Learner):
    """Kernel perceptron learner"""
    
    def __init__(self, max_iter = 100, name = "KernelPerceptron", kernel = LinealKernel()):
        self.max_iter = max_iter
        self.name = name
        self.learners = []
        self.sv_x = []
        self.sv_alpha = []
        self.kernel = kernel
        self.GRAM = None
        
    # Use orange to load the data
    def __call__(self,data,weight=0):
        """Learn from the given table of data instances."""
        
        examples = Orange.core.Preprocessor_dropMissingClasses(data)
        class_var = examples.domain.class_var
        if len(examples) == 0:
            raise ValueError("Example table is without any defined classes")
        
        # Convert the data to numpy
        X, Y, _ = data.to_numpy()
        # Add the bias
        X = np.hstack( (np.ones((X.shape[0], 1)), X))        
        # Amount of learners, base on the amount of classes
        n_learners = len(class_var.values)

        # Define the learner for each class
        # Class 1 = 0 = 'us', Class 2 = 1 = 'mw', Class 3 = 2 = 'rr'
        for i_learner in range(n_learners):
            Yclasses = np.zeros(Y.shape)
            for j in range(len(Y)):
                Yclasses[j] = 1 if Y[j] == i_learner else -1

            self.iters = 0
            
            kernelPerceptron = KernelPerceptron(self.kernel, self.GRAM)
            sv_x, sv_y, sv_alpha = None, None, None

            for iteration in range(self.max_iter):
                self.iters += 1 
                sv_x, sv_y, sv_alpha, K = kernelPerceptron.learn((X, Yclasses),(sv_x, sv_y, sv_alpha))
                if K  == 0:
                    break

            self.learners.append(kernelPerceptron)
            self.sv_x.append(sv_x)
            self.sv_alpha.append(sv_alpha*sv_y)
            Yclasses = None

            # saves the GRAM matrix
            self.GRAM = kernelPerceptron.GRAM
        X = None
        Y = None
        classifier = KernelPerceptronClassifier(predict = self.predict, domain = data.domain)
        return classifier
    
    # Multiclass Prediction
    def predict (self, X):
        n_learners = len(self.learners)
        results = []
        for i in range(n_learners):
            results.append( self.learners[i].predict(X, (self.sv_x[i], self.sv_alpha[i])))
        return results

    # Point variables to None to free up resources
    def dispose(self):
        self.sv_alpha = None
        self.sv_x = None
        self.GRAM = None
        self.kernel = None
        for learner in self.learners:
            learner.GRAM = None
            learner.kernel = None
            learner.sv_alpha = None
        self.learners =None
                
class KernelPerceptronClassifier:
    """Kernel perceptron classifier"""

    def __init__(self,**kwargs):
        self.__dict__.update(**kwargs)

    def __call__(self,example, result_type = Orange.core.GetValue):
        input = np.array([[example[feature.name].value for feature in example.domain.features]])
        results = self.predict(input)#, (self.sv_x, self.sv_y, self.sv_alpha))
        #Gets the index (class) of the results
        indexResult = self.getIndexResult(results)
        # Selecting the Majority
        mt_value =  self.domain.class_var[indexResult]
        # Set the frequency using the number of classes
        n_classes = len(results)
        freq = [0.0]*n_classes
        # Set one the class that was classified
        freq[indexResult] = 1.0
        frecuencies = freq
        mt_prob = Orange.statistics.distribution.Discrete(frecuencies)
        mt_prob.normalize()

        if result_type == Orange.core.GetValue: 
            return tuple(mt_value) if self.domain.class_vars else mt_value
        elif result_type == Orange.core.GetProbabilities: 
            return tuple(mt_prob) if self.domain.class_vars else mt_prob
        else: 
            return [tuple(mt_value), tuple(mt_prob)] if self.domain.class_vars else [mt_value, mt_prob]
        return

    def getIndexResult(self, results):
        n_classes = len(results)
        # Multi Class Classification
        flags = results > np.array([0])
        majorityFlags = flags[:,0]
        # Get Indexes that have a 1 classification
        majorityClassIndexes = np.arange( n_classes )[majorityFlags[:,0]]
        
        #Select the Max in case of a draw
        maxValue = 0
        indexResult = -1
        
        for i in range(len(majorityClassIndexes)):
            #It gets the second value of the results
            r = results[majorityClassIndexes[i]].reshape(2)[1]
            if r > maxValue:
                indexResult = majorityClassIndexes[i]
                maxValue = r
        #Select the one with lower value
        if indexResult == -1:        
            maxValue = 1
            for i in range(n_classes):
                #It gets the second value of the results
                r = results[i].reshape(2)[1]
                if r <= maxValue or maxValue == 1:
                    indexResult = i
                    maxValue = r
        return indexResult