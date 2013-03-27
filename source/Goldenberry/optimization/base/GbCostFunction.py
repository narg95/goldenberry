import abc
import ast
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
            self.acc_var += cost*cost
            

        return costs

    def statistics(self):
        """Provides the evaluations statistics in a tuple in the following order:
           evals, argmin, argmax, min, max, mean, var
        """
        if self.evals  == 0:
            raise Exception("There are no statistics for this cost function.")
        
        mean = self.acc_mean / float(self.evals)
        var = self.acc_var / float(self.evals) - mean*mean
        return self.evals, self.argmin, self.argmax, self.min, self.max, mean, var

    def reset_statistics(self):
        self.evals = 0
        self.argmin = 0
        self.argmax = 0
        self.min = float("Inf")
        self.max = float("-Inf")
        self.acc_mean = 0.0
        self.acc_var = 0.0
    
    def set_func_script(self, script):
        func_name = self.get_func_name(script)
        exec(script)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
        exec("self._cost_ = types.MethodType(" + func_name + ", self, type(self))")
    
    @staticmethod
    def get_func_name(script):
        parse_tree = ast.parse(script)
        func_def = GbCostFunction.next(ast, parse_tree, ast.FunctionDef)
        args_def = GbCostFunction.next(ast, parse_tree, ast.arguments)
        if len(args_def.args) != 2 or args_def.args[0].id != "self" and args_def.args[1].id != "solution":
            raise Exception("Cost function must have two arguments with the following names: [self, solution]")
        
        return func_def.name

    @staticmethod
    def next(ast, node, node_type):
        for node in ast.walk(node):
            if type(node) == node_type:
                return node
        raise Exception("Script is not in a correct format.")
    def set_func(self, func):
        self._cost_ = types.MethodType(func, self, type(self))

    