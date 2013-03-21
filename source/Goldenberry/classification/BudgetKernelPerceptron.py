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

class BudgetKernelPerceptron:
    """ Budget Kernel Perceptron Algorithm"""

    #Constructor
    def __init__(self, kernel, GRAM, budget = 100):
        self.kernel = kernel
        self.sv_alpha = None
        self.GRAM = GRAM
        # budget variables
        self.budget = budget
        self.budgetSet = set()
        self.budgetArray = None

    # We assume that the classes for  y are 1 and -1.
    def learn(self, (X, Y), (sv_x, sv_y, sv_alpha) = (None, None, None)):
        n_samples, n_features = X.shape
        K = 0

        # Alpha, the size is the amount of samples
        if self.sv_alpha is None:
            self.sv_alpha = np.zeros(n_samples)
        if self.budgetArray is None:
            self.budgetArray = np.zeros(n_samples)

        # If Gram matrix is not calculated, then calculated.
        # Watch out to free up the resources. Set the variables to None
        # in case of Memory Errors
        if self.GRAM is None:
            self.GRAM = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(n_samples):
                    self.GRAM[i,j] = self.kernel.compute(X[i], X[j])

        
        # Learn each sample. Increase the alpha in case the sample is missclasified.
        # Keep count of the errors (K)        
        for i in range(n_samples):
            self.budgetArray[:] = 0
            self.budgetArray[list(self.budgetSet)] = 1
            # We multiply by budget array, which is an array of 0 or 1. This works like a flag
            # to know if we should use this vector or not. This way, we keep control to only use
            # the vectors of the budget
            z = np.sign(np.sum(self.GRAM[:,i] * self.sv_alpha * self.budgetArray * Y ))
            if z != Y[i]:
                self.sv_alpha[i] += 1.0
                K += 1
                #Check if the current vector is already in the budget
                if self.budgetSet.issuperset([i]) is False:
                    if len(self.budgetSet) == self.budget:
                        #indexToRemove = self.getIndexWithMinimumCostFromBudget(i, Y)
                        indexToRemove = self.getIndexWithMinimumCostFromBudgetApproximation(Y)
                        self.budgetSet.remove(indexToRemove)
                    self.budgetSet.add(i)

        # return the data, weights and classes of the samples that make mistakes
        # and are in the budget set
        sv_x = X[list(self.budgetSet)]
        sv_y = Y[list(self.budgetSet)]
        sv_alpha = self.sv_alpha[list(self.budgetSet)]
        return sv_x, sv_y, sv_alpha, K

    # Returs the index which has the minimum cost
    # Compares the support vectors to each sample of the data set
    # and selects the one with minimum error
    def getIndexWithMinimumCostFromBudget(self, currentIndex, Y):
        flagsArray = np.zeros(len(Y))
        totalCount = -1
        minimumIndex = -1
        for budgetIndex in self.budgetSet:
            newSet = self.budgetSet.difference([budgetIndex])
            flagsArray[:] = 0
            flagsArray[list(newSet)] = 1
            count = 0
            for i in range(currentIndex):
                # Loss function is to count the errors when there is a missclasification
                if np.sign(np.sum(self.GRAM[:,i] * self.sv_alpha * flagsArray * Y)) != Y[i]:
                    count += 1
            if totalCount == -1 or count < totalCount:
                totalCount = count
                minimumIndex = budgetIndex      
        flagsArray = None     
        return minimumIndex

    # returns the index which has the minimum cost. 
    # Same approach as: getIndexWithMinimumCostFromBudget
    # approximation: It only compares the samples in the budget
    def getIndexWithMinimumCostFromBudgetApproximation(self, Y):
        flagsArray = np.zeros(len(Y))
        totalCount = -1
        minimumIndex = -1
        for budgetIndex in self.budgetSet:
            newSet = self.budgetSet.difference([budgetIndex])
            flagsArray[:] = 0
            flagsArray[list(newSet)] = 1
            count = 0
            for i in self.budgetSet:
                # Loss function is to count the errors when there is a missclasification
                if np.sign(np.sum(self.GRAM[:,i] * self.sv_alpha * flagsArray * Y)) != Y[i]:
                    count += 1
            if totalCount == -1 or count < totalCount:
                totalCount = count
                minimumIndex = budgetIndex      
        flagsArray = None     
        return minimumIndex

    # Predict the class
    # sv_alpha: already multiply the class with the weight
    def predict(self, X, (sv_x, sv_alpha)):
        n_samples = X.shape[0]
        #Adjust the size of the data to classify in case it has not the bias.
        if (X.shape[1] + 1) == sv_x.shape[1] :
            X =  np.hstack( (np.ones( (n_samples, 1)), X))

        predictions = np.zeros((n_samples,2))        
        for i in range(n_samples):        
            sum = 0        
            for aj, xj in zip(sv_alpha, sv_x):
                sum += aj * self.kernel.compute(xj, X[i])
            predictions[i] = (np.sign(sum), sum)
        
        #return a tuple with the sign and the distance to the hyperplane
        return predictions

class BudgetKernelPerceptronLearner(Learner):
    """Kernel perceptron learner"""
    
    def __init__(self, max_iter = 1, name = "BudgetKernelPerceptron", kernel = LinealKernel(), budget = 100):
        self.max_iter = max_iter
        self.name = name
        self.learners = []
        self.sv_x = []
        self.sv_alpha = []
        self.kernel = kernel
        self.budget = budget
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
            # Set the class that being is learned to 1 and the others
            # to -1. One vs against all.
            for j in range(len(Y)):
                Yclasses[j] = 1 if Y[j] == i_learner else -1
                
            self.iters = 0
            
            kernelPerceptron = BudgetKernelPerceptron(self.kernel, self.GRAM, budget = self.budget)
            sv_x, sv_y, sv_alpha = None, None, None

            for iteration in range(self.max_iter):
                self.iters += 1 
                sv_x, sv_y, sv_alpha, K = kernelPerceptron.learn((X, Yclasses),(sv_x, sv_y, sv_alpha))
                if K  == 0:
                    break

            self.learners.append(kernelPerceptron)
            self.sv_x.append(sv_x)
            alphas = sv_alpha*sv_y
            self.sv_alpha.append(alphas)
            Yclasses = None
            
            # saves the GRAM matrix
            self.GRAM = kernelPerceptron.GRAM

        X = None
        Y = None
        classifier = BudgetKernelPerceptronClassifier(predict = self.predict, domain = data.domain)
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
            learner.budgetSet = None
            learner.budgetArray = None
        self.learners =None
                    
class BudgetKernelPerceptronClassifier:
    """Budget Kernel perceptron learner"""

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