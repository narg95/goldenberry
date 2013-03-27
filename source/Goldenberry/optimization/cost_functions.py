import numpy as np

def OneMax(self, solution):
    """It calculates the total number of ones in a vector."""
    return solution.sum()

def LeadingOnes(self, solution):
    """it counts the number of ones in your vector, starting at the beginning, until a zero is encountered."""
    loc = numpy.where(solution == 0.0)[0]
    if len(loc) > 0:
        return loc[0]
    return len(solution)

def LeadingOnesBlocks(self, solution):
    """Given a block size, we count the number of strings of ones until a zero is found."""
    block_size = 1
    loc = numpy.where(solution == 0.0)[0]
    if len(loc) > 0:
        return loc[0]
    return len(solution)

def Traps(self, solution):
    """it counts the number of ones in your vector, starting at the beginning, until a zero is encountered."""
    if solution.all():
        return len(solution) + 1.0
    return len(np.nonzero(solution == 0.0))

def ZeroMax(self, solution):
    """It calculates the total number of zeros in a vector."""
    return len(solution) - solution.sum()

def Linear(self, solution):
    """It calculates the total number of zeros in a vector."""
    return len(solution) - solution.sum()

#class ZeromaxTruncated(GbCostFunction):
#    """ Zero cost"""
#    def __cost__(self, solution):
#        return len(solution) - np.minimum(np.ones(len(solution)), np.fabs(solution)).sum()


#class CondOnemax(GbCostFunction):
#    """ Conditional onemax cost function"""
    
#    def __cost__(self, solution):
#        prev = 1.0
#        cost = 0.0
#        cond_indexes = range(len(solution))

#        for i in cond_indexes:
#            curr =  solution[i]
#            cost += prev * curr
#            prev = curr
        
#        return cost
            
