import numpy as np
import types
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

class Custom(GbBaseCostFunction):
    
    def __init__(self, script):
        super(Custom, self).__init__()
        self.set_func_script(script)
    
    def set_func_script(self, script):
        exec("def __cost_func__(self, solution):\n\t" + script.replace("\n","\n\t"))
        self.__custom_cost__ = types.MethodType(__cost_func__, self, type(self))

    def __cost__(self, solution):
        if hasattr(self, '__custom_cost__'):
            return self.__custom_cost__(solution)
        return 0.0
            
