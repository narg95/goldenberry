import abc
import sys
import numpy as np
class GbBaseCostFunction:
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.reset_statistics()

    """This is the base class for all the cost functions"""
    def __call__(self, solutions):
        return self.cost(solutions)

    def reset_statistics(self):
        self.evals = 0
        self.argmin = 0
        self.argmax = 0
        self.min = float("Inf")
        self.max = float("-Inf")
        self.acc_mean = 0.0
        self.acc_stdev = 0.0
        

    def cost(self, solutions):
        """Gets the cost of a given solution and calculate the statistics."""
        costs = np.zeros(solutions.shape[0])
        
        for idx, solution in enumerate(solutions):
            self.evals += 1    
            cost = self.__cost__(solution)
            
            if cost > self.max:
                self.max = cost
                self.argmax = self.evals
            
            elif cost < self.min:
                self.min = cost
                self.argmin = self.evals

            costs[idx] = cost
            self.acc_mean += cost
            self.acc_stdev += cost*cost
            

        return costs

    @abc.abstractmethod
    def __cost__(self, solution):
        """Calculate the cost for the given solution."""
        pass
        
    def statistics(self):
        """Provides the evaluations statistics in a tuple in the following order:
           evals, argmin, argmax, min, max, mean, stdev
        """
        if self.evals  == 0:
            raise Exception("There are no statistics for this cost function.")
        
        mean = self.acc_mean / float(self.evals)
        stdev = self.acc_stdev / float(self.evals) - mean*mean
        return self.evals, self.argmin, self.argmax, self.min, self.max, mean, stdev
