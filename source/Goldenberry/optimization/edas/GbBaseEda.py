import abc
import numpy as np
from Goldenberry.optimization.base.GbBaseOptimizer import GbBaseOptimizer
from Goldenberry.optimization.base.GbSolution import GbSolution

class GbBaseEda(GbBaseOptimizer):
    """This class represents a solution with a cost associated."""
    __metaclass__ = abc.ABCMeta

    distr = None
    cand_size = None
    var_size = None
    max_iters = None
    max_evals = None
    percentage = None
    learning_rate = None

    def setup(self, var_size = 10, cand_size = 20, max_iters = None, max_evals = None, percentage = 50, learning_rate = 1.0, **kwargs):
        """Configure a Eda instance"""
        self.cand_size = cand_size
        self.var_size = var_size
        self.max_iters = max_iters
        self.max_evals = max_evals
        self.percentage = percentage
        self.learning_rate = learning_rate
        self.__dict__.update(**kwargs)

        self.reset()
        self.initialize()

    def reset(self):
        self.iters = 0
        if None != self.cost_func:
            self.cost_func.reset_statistics()
        if None != self.distr:
            self.distr.reset()            
        
    @abc.abstractmethod
    def initialize(self):
        """Initializes algorithm's specific settings like the probability distribution."""
        raise NotImplementedError()

    def ready(self):
        """"Checks whether the algorithm is ready or not for executing."""
        return (self.cost_func != None \
                and None != self.cand_size \
                and None != self.var_size\
                and None != self.distr \
                and self.cand_size > 0 \
                and self.var_size > 0 
                and self.percentage > 0 \
                and self.percentage < 100 \
                and self.percentage > 0)

    def search(self):
        """Search for an optimal solution."""        
        best_one = GbSolution(None, float("-Inf"))
        
        best = None
        while not self.hasFinished():
            candidates = self.generate_candidates(self.cand_size, best)
            best, winner = self.best_candidates(candidates)
            self.update_distribution(best, best_one)
            
            if best_one.cost < winner.cost:
                best_one = winner
            
            self.iters += 1

        return best_one

    def hasFinished(self):
        finish = (not (self.max_iters is None) and self.iters > self.max_iters) or \
                 (not (self.max_evals is None) and self.cost_func.evals > self.max_evals)
        
        if finish:
            return True
        return self.distr.has_converged()

    def generate_candidates(self, sample_size, best):
        """Generates the new generation of candidate solutions."""
        return self.distr.sample(sample_size)

    def best_candidates(self, candidates):
        fits = self.cost_func(candidates)
        index = np.argsort(fits)[:(self.cand_size * self.percentage/100):-1]
        return candidates[index], GbSolution(candidates[index[0]], fits[index[0]])

    @abc.abstractmethod
    def update_distribution(self, candidates, best_one):
        """Updates the current distribution."""
        raise NotImplementedError()

    
  
    