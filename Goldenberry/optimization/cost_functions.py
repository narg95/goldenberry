import numpy as np

def OneMax(solution):
    """Computes the total number of ones in the candidate solution."""
    return solution.sum()

def LeadingOnes(solution):
    """Counts the number of ones in the candidate solution, starting at the beginning, until a zero is encountered."""
    loc = numpy.where(solution == 0.0)[0]
    if len(loc) > 0:
        return loc[0]
    return len(solution)

def LeadingOnesBlocks(solution):
    """Given a block size, counts the number of strings of ones until a zero is found."""
    block_size = 3
    score = 0.0
    for idx in range(len(solution)/block_size):
        score += solution[(idx)*block_size:(idx + 1)*block_size].prod()    
    return score

def Traps(solution):
    """it counts the number of ones in your vector, starting at the beginning, until a zero is encountered."""
    if solution.all():
        return len(solution) + 1.0
    return len(np.nonzero(solution == 0.0))

def ZeroMax(solution):
    """It calculates the total number of zeros in a vector."""
    return len(solution) - solution.sum()

def Linear(solution):
    """It calculates the total number of zeros in a vector."""
    return len(solution) - solution.sum()
