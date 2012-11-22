import numpy as np
from Goldenberry.optimization.edas.distributions import *
from Goldenberry.optimization.edas.BaseEda import BaseEda

class Cga(BaseEda):
    """Compact Genetic Algorithm"""
    
    def setup(self, cost_function, varsize, popsize, maxiters = None):
        self._pop_size = popsize
        self._vars_size = varsize
        self._cost_function = cost_function
        self._distribution = Binomial(parameters = np.tile(0.5,(1,varsize)))
        self._max_iters = maxiters
        self._iters = 0

    def search(self):
        while not self.hasFinished():
            self._iters += 1
            pop = self._distribution.sample(self._distribution, 2, self._vars_size)
            winner, losser = self.compete(pop)
            self.estimate_distribution(winner, losser)
        
        return self._distribution.sample(1)

    def ready(self):
        return self._pop_size is not None and\
               self._vars_size is not None and\
               self._cost_function is not None

    def hasFinished(self):
        finish = not (self._max_iters is None) and self._iters > self._max_iters
        if finish:
            return True
        return (((1 - self._distribution()) < 0.01) | (self._distribution() < 0.01)).all()
    
    def compete(self, pop):
        minindx = argmin(self._cost_function(pop))
        maxindx = 1 if minindx == 0 else 0
        return  pop[maxindx], pop[minindx]

    def estimate_distribution(self, winner, losser):
        for x in range(0, self._vars_size):
            if 1.0 >= self._distribution[0,x] >= 0.0:
                self._distribution[0,x] += (winner[x] - losser[x])/float(self._pop_size)

    def result_distribution(self):
        """Provides the final estimated distribution."""
        return self._distribution