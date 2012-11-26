from numpy import *

def sample(probs, popsize, varsize):
    return (dot(ones((popsize,1)), probs ) >= random.rand(popsize, varsize)) + 0