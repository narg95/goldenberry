import numpy as np
from Goldenberry.statistics.distributions import *
from Goldenberry.optimization.edas.GbBaseEda import GbBaseEda
from Goldenberry.optimization.base.GbSolution import *

class Cga(GbBaseEda):
    """Compact Genetic Algorithm"""

    cand_size = None
    var_size = None
    distr = None
    max_iters = None
    iter = None
    max_evals = None

    def setup(self, var_size, cand_size, max_iters = None, max_evals = None):
        """Configure a Cga instance"""
        self.cand_size = cand_size
        self.var_size = var_size
        self.max_iters = max_iters
        self.max_evals = max_evals

        self.reset()

    def reset(self):
        self.iter = 0
        if None != self.var_size:
            self.distr = Binomial(n = self.var_size)    
        if None != self.cost_func:
            self.cost_func.reset_statistics()

    def update_candidates(self):
        """Generates the new pair of candidates"""
        return self.distr.sample(2)

    def search(self):
        """Search for an optimal solution."""        
        best_candidate = GbSolution(None, 0.0)
        
        while not self.hasFinished():
            candidates = self.update_candidates()
            winner, losser = self.compete(candidates)
            self.update_distribution(winner, losser)
            
            if best_candidate.cost < winner.cost:
                best_candidate = winner
            
            self.iter += 1

        #returns the best candidate found so far
        return best_candidate

    def ready(self):
        """"Checks whether the algorithm is ready or not for executing."""
        return self.cand_size is not None and\
               self.var_size is not None and\
               super(Cga, self).ready() 

    def hasFinished(self):
        finish = (not (self.max_iters is None) and self.iter > self.max_iters) or \
                 (not (self.max_evals is None) and self.cost_func.evals > self.max_evals)
        
        if finish:
            return True
        return (((1 - self.distr()) < 0.01) | (self.distr() < 0.01)).all()
    
    def compete(self, candidates):
        costs = self.cost_func(candidates)
        maxindx = np.argmax(costs)
        return  GbSolution(candidates[maxindx], costs[maxindx]), GbSolution(candidates[not maxindx], costs[not maxindx])

    def update_distribution(self, winner, losser):
        self.distr.p = np.minimum(np.ones((1, self.var_size)), np.maximum(np.zeros((1, self.var_size)), self.distr.p + (winner.params - losser.params) / float(self.cand_size)))

    @property
    def distribution(self):
        """Provides the final estimated distribution."""
        return self.distr