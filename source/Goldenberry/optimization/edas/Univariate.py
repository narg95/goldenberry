import numpy as np
from Goldenberry.statistics.distributions import Binomial, GaussianTrunc
from Goldenberry.optimization.edas.GbBaseEda import GbBaseEda
from Goldenberry.optimization.base.GbSolution import GbSolution

class Cga(GbBaseEda):
    """Compact Genetic Algorithm"""

    def initialize(self):
        self.distr = Binomial(self.var_size)

    def setup(self, cand_size = 20, max_evals = 100, selection_rate = 50, learning_rate = 1, callback_func = None, **kwargs):
        super(Cga, self).setup(cand_size, max_evals, selection_rate, learning_rate, callback_func, **kwargs)
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
        self.distr.p =  self.distr.p*(1-self.learning_rate) + self.learning_rate * np.average(top_ranked, axis = 0)

class Tilda(GbBaseEda):
    """Tilda algorithm."""

    low = 0.0
    high = 1.0
        
    def initialize(self):
        self.distr = GaussianTrunc(n = self.var_size, low = self.low, high = self.high)
        self.acc_mean = np.zeros(self.var_size)
        self.acc_vars = np.zeros(self.var_size)        
    
    def sample(self, sample_size, top_ranked, best):
        """Generates the new pair of candidates"""
        return self.distr.sample(2)

    def estimate(self, winner, best):
        self.acc_mean += winner.params
        self.acc_vars += winner.params*winner.params        
        if self.iters % (self.cand_size/2) == 0:
            means, stds = \
                Tilda._estimate_gaussian(\
                    self.distr.means, \
                    self.distr.stdevs, \
                    self.acc_mean, \
                    self.acc_vars, \
                    best, \
                    self.cand_size, \
                    self.learning_rate)
            self.distr.means = means
            self.distr.stdevs = stds
            self.acc_mean = np.zeros(self.var_size)
            self.acc_vars = np.zeros(self.var_size)

    def get_top_ranked(self, candidates):
        costs = self.cost_func(candidates)
        maxindx = np.argmax(costs)
        winner, loser = GbSolution(candidates[maxindx], costs[maxindx]), GbSolution(candidates[not maxindx], costs[not maxindx])
        return  winner, winner

    @staticmethod
    def _estimate_gaussian(means, stds, acc_means, acc_vars, best, cand_size, learning_rate):
        acc_means = acc_means/float(cand_size/2.0)
        acc_std = np.sqrt((acc_vars/float(cand_size/2.0)) - np.square(acc_means))
        #if None is not best.params:
        #    means = means*(1.0 - learning_rate) + \
        #                        learning_rate*((acc_means + best.params) / 2.0)
        #else:
        means = means*(1.0 - learning_rate) + \
                                learning_rate * (acc_means)

        stds = stds * (1-learning_rate) + acc_std * learning_rate
        return means, stds

class Pbilc(GbBaseEda):
    """PBILc algorithm."""

    low = 0.0
    high = 1.0
    
    def initialize(self):
        self.distr = GaussianTrunc(n = self.var_size, low = self.low, high = self.high)
    
    def estimate(self, top_ranked, best):
        self.distr.means = self.distr.means * (1 - self.learning_rate) + np.average(top_ranked, axis = 0) * self.learning_rate
        self.distr.stdevs = self.distr.stdevs * (1 - self.learning_rate) + np.std(top_ranked, axis = 0) * self.learning_rate