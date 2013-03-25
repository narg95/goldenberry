import numpy as np
from Goldenberry.statistics.distributions import Binomial, Gaussian
from Goldenberry.optimization.edas.GbBaseEda import GbBaseEda
from Goldenberry.optimization.base.GbSolution import GbSolution

class Cga(GbBaseEda):
    """Compact Genetic Algorithm"""

    def initialize(self):
        self.distr = Binomial(self.var_size)
        self.learning_rate = 1.0/float(self.cand_size)
        self.sample_size = 2
    
    def get_top_ranked(self, candidates):
        costs = self.cost_func(candidates)
        maxindx = np.argmax(costs)
        winner, loser = GbSolution(candidates[maxindx], costs[maxindx]), GbSolution(candidates[not maxindx], costs[not maxindx])
        return  (winner, loser), winner

    def estimate(self, (winner, loser), best):
        self.distr.p = np.minimum(np.ones((1, self.var_size)), np.maximum(np.zeros((1, self.var_size)), self.distr.p + (winner.params - loser.params) * self.learning_rate))
   
class Pbil(GbBaseEda):
    """Pbil Algorithm"""

    def initialize(self):
        self.distr = Binomial(self.var_size)    

    def estimate(self, top_ranked, best):
        self.distr.p =  self.distr.p*(1-self.learning_rate) + self.learning_rate * np.average(top_ranked)

class Tilda(GbBaseEda):
    """Tilda algorithm."""

    low = 0.0
    high = 1.0
    
    def initialize(self):
        self.distr = Gaussian(n = self.var_size)
        self.acc_mean = np.zeros(self.var_size)
        self.acc_vars = np.zeros(self.var_size)
    
    def sample(self, sample_size, top_ranked, best):
        """Generates the new pair of candidates"""
        return self.distr.sample(2)

    def estimate(self, winner, best):
        if self.iters % (self.cand_size/2) != 0:
            self.acc_mean += winner.params
            self.acc_vars += winner.params*winner.params
        else:
            means, vars = \
                Tilda._estimate_gaussian(\
                    self.distr.means, \
                    self.distr.stdevs * self.distr.stdevs, \
                    self.acc_mean, \
                    self.acc_vars, \
                    best, \
                    self.cand_size, \
                    self.learning_rate)
            self.distr.means = means
            self.distr.stdevs = np.sqrt(vars)
            self.acc_mean = np.zeros(self.var_size)
            self.acc_vars = np.zeros(self.var_size)

    def get_top_ranked(self, candidates):
        costs = self.cost_func(candidates)
        maxindx = np.argmax(costs)
        winner, loser = GbSolution(candidates[maxindx], costs[maxindx]), GbSolution(candidates[not maxindx], costs[not maxindx])
        return  winner, winner

    @staticmethod
    def _estimate_gaussian(means, vars, acc_means, acc_vars, best, cand_size, learning_rate):
        acc_means = acc_means/float(cand_size)
        acc_vars = acc_vars/float(cand_size)
        if None != best.params:
            means = means*(1.0 - learning_rate) + \
                                learning_rate*((acc_means + best.params) / 2.0)
        else:
            means = means*(1.0 - learning_rate) + \
                                learning_rate * (acc_means)

        vars = vars*(1-learning_rate) + (acc_vars - acc_means*acc_means)*learning_rate            
        return means, vars