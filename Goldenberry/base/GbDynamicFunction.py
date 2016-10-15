import abc
import ast
import sys
import numpy as np
import inspect
import types

class GbDynamicFunction:
    __metaclass__ = abc.ABCMeta

    def __init__(self, func = None, script = None):
        self.reset_statistics()
        if func is not None:
            self.set_func(func)
        elif script is not None:
            self.set_func_script(script)
        else:
            raise Exception("Please provide either a function or function script.")

    def __call__(self, *args, **kwargs):
        """This is the base class for all the cost functions"""
        return self.execute(*args, **kwargs)

    def execute(self, *args, **kwargs):
        result = self._dynamic_function_(*args, **kwargs)
        self._update_statistics(result)
        return result

    def _update_statistics(self, result):
        self.evals += 1
        if result > self.max:
            self.max = result
            self.argmax = self.evals
            
        elif result < self.min:
            self.min = result
            self.argmin = self.evals

        self.acc_mean += result
        self.acc_var += result * result

    def statistics(self):
        """Provides the evaluations statistics in a tuple in the following order:
           evals, argmin, argmax, min, max, mean, var.
        """
        if self.evals  == 0:
            raise Exception("There are no statistics for this cost function.")
        
        mean = self.acc_mean / float(self.evals)
        var = self.acc_var / float(self.evals) - mean*mean
        return self.evals, self.argmin, self.argmax, self.min, self.max, mean, np.sqrt(var)

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
        #exec("self._dynamic_function_ = types.MethodType(" + func_name + ", self, type(self))")
        exec("self._dynamic_function_ = " + func_name )
    
    @staticmethod
    def get_func_name(script):
        parse_tree = ast.parse(script)
        func_def = GbDynamicFunction.next(ast, parse_tree, ast.FunctionDef)
        args_def = GbDynamicFunction.next(ast, parse_tree, ast.arguments)
        if len(args_def.args) < 1 and args_def.args[0].id != "self":
            raise Exception("Custom function must have at least the 'self' parameter.")
        
        return func_def.name

    @staticmethod
    def next(ast, node, node_type):
        for node in ast.walk(node):
            if type(node) == node_type:
                return node
        raise Exception("Script is not in a correct format.")
    def set_func(self, func):
        self._dynamic_function_ = func