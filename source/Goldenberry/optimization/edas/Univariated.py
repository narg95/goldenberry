import numpy as np
from Goldenberry.statistics.distributions import Binomial
from Goldenberry.optimization.edas.GbBaseEda import GbBaseEda
from Goldenberry.optimization.base.GbSolution import GbSolution

class Cga(GbBaseEda):
    """Compact Genetic Algorithm"""

    def initialize(self):
        self.distr = Binomial(self.var_size)
        self.learning_rate = 1.0/float(self.cand_size)
    
    def generate_candidates(self, sample_size, best):
        """Generates the new pair of candidates"""
        return self.distr.sample(2)

    def best_candidates(self, candidates):
        costs = self.cost_func(candidates)
        maxindx = np.argmax(costs)
        winner, loser = GbSolution(candidates[maxindx], costs[maxindx]), GbSolution(candidates[not maxindx], costs[not maxindx])
        return  (winner, loser), winner

    def update_distribution(self, (winner, loser)):
        self.distr.p = np.minimum(np.ones((1, self.var_size)), np.maximum(np.zeros((1, self.var_size)), self.distr.p + (winner.params - loser.params) * self.learning_rate))
   
class Pbil(GbBaseEda):
    """Pbil Algorithm"""

    def initialize(self):
        self.distr = Binomial(self.var_size)    

    def update_distribution(self, best):
        self.distr.p = np.minimum(np.ones((1, self.var_size)), np.maximum(np.zeros((1, self.var_size)), self.distr.p*(1-self.learning_rate) + self.learning_rate * np.average(best) ))
