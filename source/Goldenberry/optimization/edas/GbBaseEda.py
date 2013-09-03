import abc
import numpy as np
from Goldenberry.optimization.base.GbBaseOptimizer import GbBaseOptimizer
from Goldenberry.optimization.base.GbSolution import GbSolution

class GbBaseEda(GbBaseOptimizer):
    """This class represents a base class for Estimation of distribution algorithm."""
    __metaclass__ = abc.ABCMeta

    distr = None
    cand_size = None
    sample_size = None
    var_size = None
    max_iters = None
    max_evals = None
    selection_rate = None
    learning_rate = None

    def setup(self, var_size = 10, cand_size = 20, max_iters = None, max_evals = None, selection_rate = 50, learning_rate = 1.0, **kwargs):
        """Configure a Eda instance"""
        self.cand_size = cand_size
        self.sample_size = cand_size
        self.var_size = var_size
        self.max_iters = max_iters
        self.max_evals = max_evals
        self.selection_rate = selection_rate
        self.learning_rate = learning_rate
        self.__dict__.update(**kwargs)

        self.reset()
        self.initialize()

    def reset(self):
        self.iters = 0
        if None is not self.cost_func:
            self.cost_func.reset_statistics()
        if None is not self.distr:
            self.distr.reset()            
        
    @abc.abstractmethod
    def initialize(self):
        """Initializes algorithm's specific settings like the probability distribution."""
        raise NotImplementedError()

    def ready(self):
        """"Checks whether the algorithm is ready or not for executing."""
        return (self.cost_func is not None \
                and None is not self.cand_size \
                and None is not self.var_size\
                and None is not self.distr \
                and self.cand_size > 0 \
                and self.var_size > 0 
                and self.selection_rate > 0 \
                and self.selection_rate < 100 \
                and self.selection_rate > 0)

    def search(self):
        """Search for an optimal solution."""        
        if not self.ready():
            raise Exception("The optimizer is not ready for being executed.  Please check if you have configured all the required parameters.")
        best = GbSolution(None, float('-Inf'))
        top_ranked = None
        while not self.done():
            self.iters += 1
            candidates = self.sample(self.sample_size, top_ranked, best)
            top_ranked, winner = self.get_top_ranked(candidates)
            self.estimate(top_ranked, best)
            
            if best.cost < winner.cost:
                best = winner
            
        return best

    def done(self):
        finish = (not (self.max_iters is None) and self.iters > self.max_iters) or \
                 (not (self.max_evals is None) and self.cost_func.evals > self.max_evals)
        
        if finish:
            return True
        return self.distr.has_converged()

    def sample(self, sample_size, top_ranked, best):
        """Generates the new generation of candidate solutions."""
        return self.distr.sample(sample_size)

    def get_top_ranked(self, candidates):
        fits = self.cost_func(candidates)
        index = np.argsort(fits)[:(self.cand_size * self.selection_rate/100):-1]
        return candidates[index], GbSolution(candidates[index[0]], fits[index[0]])

    @abc.abstractmethod
    def estimate(self, top_ranked, best):
        """Updates the current distribution."""
        raise NotImplementedError()

    
  
    