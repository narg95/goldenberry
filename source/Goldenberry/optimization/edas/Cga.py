import numpy as np
from Goldenberry.statistics.distributions import *
from Goldenberry.optimization.edas.BaseEda import BaseEda
from Goldenberry.optimization.base.GbSolution import *

class Cga(BaseEda):
    """Compact Genetic Algorithm"""

    cand_size = None
    var_size = None
    cost_func = None
    distr = None
    max_iters = None
    iter = None
    fit_evals = None

    def setup(self, cost_function, var_size, cand_size, max_iters = None):
        """Configure a Cga instance"""
        self.cand_size = cand_size
        self.var_size = var_size
        self.cost_func = cost_function
        self.distr = Binomial(n = var_size)
        self.max_iters = max_iters
        self.iter = 0
        self.fit_evals = 0

    def update_candidates(self):
        """Generates the new pair of candidates"""
        return self.distr.sample(2)

    def search(self):
        """Search for an optimal solution."""        
        best_candidate = GbSolution(None, 0.0)
        
        while not self.hasFinished():
            pop = self.update_candidates()
            winner, losser = self.compete(pop)
            self.update_distribution(winner, losser)
            
            if best_candidate.cost < winner.cost:
                best_candidate = winner
            
            self.fit_evals += 2
            self.iter += 1

        #returns the best candidate found so far
        return best_candidate

    def ready(self):
        """"Checks whether the algorithm is ready or not for executiing."""
        return self.cand_size is not None and\
               self.var_size is not None and\
               self.cost_func is not None

    def hasFinished(self):
        finish = not (self.max_iters is None) and self.iter > self.max_iters
        if finish:
            return True
        return (((1 - self.distr()) < 0.01) | (self.distr() < 0.01)).all()
    
    def compete(self, pop):
        costs = self.cost_func(pop)
        maxindx = np.argmax(costs)
        return  GbSolution(pop[maxindx], costs[maxindx]), GbSolution(pop[not maxindx], costs[not maxindx])

    def update_distribution(self, winner, losser):
        self.distr.p = np.minimum(np.ones((1, self.var_size)), np.maximum(np.zeros((1, self.var_size)), self.distr.p + (winner.params - losser.params) / float(self.cand_size)))

    @property
    def distribution(self):
        """Provides the final estimated distribution."""
        return self.distr