import abc
import sys
import numpy as np
import inspect
import types

class GbCostFunction:
    __metaclass__ = abc.ABCMeta

    def __init__(self, func = None, script = None):
        self.reset_statistics()
        if func != None:
            self.set_func(func)
        elif script != None:
            self.set_func_script(script)
        else:
            raise Exception("Please provide either a function or function script.")

    def __call__(self, solutions):
        """This is the base class for all the cost functions"""
        return self.cost(solutions)

    def cost(self, solutions):
        """Gets the cost of a given solution and calculate the statistics."""
        
        costs = np.zeros(solutions.shape[0])
        
        for idx, solution in enumerate(solutions):
            self.evals += 1    
            cost = self._cost_(solution)
            
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

    def statistics(self):
        """Provides the evaluations statistics in a tuple in the following order:
           evals, argmin, argmax, min, max, mean, stdev
        """
        if self.evals  == 0:
            raise Exception("There are no statistics for this cost function.")
        
        mean = self.acc_mean / float(self.evals)
        stdev = self.acc_stdev / float(self.evals) - mean*mean
        return self.evals, self.argmin, self.argmax, self.min, self.max, mean, stdev

    def reset_statistics(self):
        self.evals = 0
        self.argmin = 0
        self.argmax = 0
        self.min = float("Inf")
        self.max = float("-Inf")
        self.acc_mean = 0.0
        self.acc_stdev = 0.0
    
    def set_func_script(self, script):
        exec("def __temp_func__(self, solution):\n\t" + script.replace("\n","\n\t"))
        self._cost_ = types.MethodType(__temp_func__, self, type(self))

    def set_func(self, func):
        self._cost_ = types.MethodType(func, self, type(self))

    