from numpy import *

def constant(f):
    """Defines an attribute/annotation for creating CONSTANTS values"""
    def fset(self, value):
        raise SyntaxError
    def fget(self):
        return f()
    return property(fget, fset)

def sample(probs, popsize, varsize):
    return (dot(ones((popsize,1)), probs ) >= random.rand(popsize, varsize)) + 0