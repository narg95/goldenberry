import numpy as np
from Goldenberry.optimization.base.GbBaseCostFunction import GbBaseCostFunction

class Onemax(GbBaseCostFunction):
    """ One max"""
    def cost(self, solutions):
        return solutions.sum(axis=1)

    def name(self):
        """Gets name"""
        return "One max"

class Zeromax(GbBaseCostFunction):
    """ Zero cost"""
    def cost(self, solutions):
        return solutions.shape[1] - solutions.sum(axis=1)

    def name(self):
        """Gets name"""
        return "Zero function"

class CondOnemax(GbBaseCostFunction):
    """ Conditional onemax cost function"""

    def cost(self, solutions):
        prev = np.ones(solutions.shape[0])
        cost = np.zeros(solutions.shape[0])
        cond_indexes = range(solutions.shape[1])

        for i in range(solutions.shape[1]):
            curr =  solutions[:,cond_indexes[i]]
            cost += prev * curr
            prev = curr
        
        return cost    


    def name(self):
        """Gets name"""
        return "Conditional onemax function"