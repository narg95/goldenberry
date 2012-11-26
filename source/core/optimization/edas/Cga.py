import numpy as np
from Goldenberry.optimization.edas.distributions import *
from Goldenberry.optimization.edas.BaseEda import BaseEda
from Goldenberry.optimization.base.GbSolution import *

class Cga(BaseEda):
    """Compact Genetic Algorithm"""
    
    def setup(self, cost_function, varsize, popsize, maxiters = None):
        """Configure a Cga instance"""
        self._pop_size = popsize
        self._vars_size = varsize
        self._cost_function = cost_function
        self._distribution = Binomial(params = np.tile(0.5,(1,varsize)))
        self._max_iters = maxiters
        self._iters = 0

    def search(self):
        """Search for an optimal solution."""
        while not self.hasFinished():
            self._iters += 1
            pop = self._distribution.sample(2)
            winner, losser = self.compete(pop)
            self.estimate_distribution(winner, losser)
        
        #returns the winner with its estimated cost
        winner = self._distribution.sample(1)
        return GbSolution(winner, self._cost_function(winner))

    def ready(self):
        """"Checks whether the algorithm is ready or not for executiing."""
        return self._pop_size is not None and\
               self._vars_size is not None and\
               self._cost_function is not None

    def hasFinished(self):
        finish = not (self._max_iters is None) and self._iters > self._max_iters
        if finish:
            return True
        return (((1 - self._distribution()) < 0.01) | (self._distribution() < 0.01)).all()
    
    def compete(self, pop):
        maxindx = bool(np.argmax(self._cost_function(pop)))
        return  pop[maxindx], pop[not maxindx]

    def estimate_distribution(self, winner, losser):
        for x in range(0, self._vars_size):
            if 1.0 >= self._distribution[0,x] >= 0.0:
                self._distribution[0,x] += (winner[0,x] - losser[0,x])/float(self._pop_size)

    def result_distribution(self):
        """Provides the final estimated distribution."""
        return self._distribution