import numpy as np
from Goldenberry.optimization.base.GbBaseCostFunction import GbBaseCostFunction

class Onemax(GbBaseCostFunction):
    """ One max"""
    def __cost__(self, solution):
        return solution.sum()

class Zeromax(GbBaseCostFunction):
    """ Zero cost"""
    def __cost__(self, solution):
        return len(solution) - solution.sum()

class CondOnemax(GbBaseCostFunction):
    """ Conditional onemax cost function"""

    def __cost__(self, solution):
        prev = 1.0
        cost = 0.0
        cond_indexes = range(len(solution))

        for i in cond_indexes:
            curr =  solution[i]
            cost += prev * curr
            prev = curr
        
        return cost