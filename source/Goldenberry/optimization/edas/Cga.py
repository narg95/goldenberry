import numpy as np
from Goldenberry.statistics.distributions import *
from Goldenberry.optimization.edas.BaseEda import BaseEda
from Goldenberry.optimization.base.GbSolution import *

class Cga(BaseEda):
    """Compact Genetic Algorithm"""

    pop_size = None
    var_size = None
    cost_func = None
    distr = None
    max_iters = None
    iter = None

    def setup(self, cost_function, varsize, popsize, maxiters = None):
        """Configure a Cga instance"""
        self.pop_size = popsize
        self.var_size = varsize
        self.cost_func = cost_function
        self.distr = Binomial(n = varsize)
        self.max_iters = maxiters
        self.iter = 0

    """Generates the new pair of candidates"""
    def generate_candidates(self):
        return self.distr.sample(2)

    def search(self):
        """Search for an optimal solution."""
        while not self.hasFinished():
            self.iter += 1
            pop = self.generate_candidates()
            winner, losser = self.compete(pop)
            self.update_distribution(winner, losser)
        
        #returns the winner with its estimated cost
        winner = self.distr.sample(1)
        return GbSolution(winner, self.cost_func(winner))

    def ready(self):
        """"Checks whether the algorithm is ready or not for executiing."""
        return self.pop_size is not None and\
               self.var_size is not None and\
               self.cost_func is not None

    def hasFinished(self):
        finish = not (self.max_iters is None) and self.iter > self.max_iters
        if finish:
            return True
        return (((1 - self.distr()) < 0.01) | (self.distr() < 0.01)).all()
    
    def compete(self, pop):
        maxindx = bool(np.argmax(self.cost_func(pop)))
        return  pop[maxindx], pop[not maxindx]

    def update_distribution(self, winner, losser):
        self.distr.p = np.minimum(np.ones((1, self.var_size)), np.maximum(np.zeros((1, self.var_size)) ,self.distr.p + (winner-losser)/float(self.pop_size)))

    @property
    def distribution(self):
        """Provides the final estimated distribution."""
        return self.distr