import numpy as np
from Goldenberry.statistics.distributions import Binomial, Gaussian
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

    def get_top_ranked(self, candidates):
        costs = self.cost_func(candidates)
        maxindx = np.argmax(costs)
        winner, loser = GbSolution(candidates[maxindx], costs[maxindx]), GbSolution(candidates[not maxindx], costs[not maxindx])
        return  (winner, loser), winner

    def estimate_distribution(self, (winner, loser), best_one):
        self.distr.p = np.minimum(np.ones((1, self.var_size)), np.maximum(np.zeros((1, self.var_size)), self.distr.p + (winner.params - loser.params) * self.learning_rate))
   
class Pbil(GbBaseEda):
    """Pbil Algorithm"""

    def initialize(self):
        self.distr = Binomial(self.var_size)    

    def estimate_distribution(self, best, best_one):
        self.distr.p =  self.distr.p*(1-self.learning_rate) + self.learning_rate * np.average(best)

class Tilda(GbBaseEda):
    """Tilda algorithm."""

    def initialize(self):
        self.distr = Gaussian(n = self.var_size)
        self.acc_mean = np.zeros(self.var_size)
        self.acc_vars = np.zeros(self.var_size)
    
    def generate_candidates(self, sample_size, best):
        """Generates the new pair of candidates"""
        return self.distr.sample(2)

    def estimate_distribution(self, winner, best_one):
        if self.iters % (self.cand_size/2) != 0:
            self.acc_mean += winner.params
            self.acc_vars += winner.params*winner.params
        else:
            means, vars = \
                Tilda.calculate_means_and_vars(\
                    self.distr.means, \
                    self.distr.stdevs * self.distr.stdevs, \
                    self.acc_mean, \
                    self.acc_vars, \
                    best_one, \
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
    def calculate_means_and_vars(means, vars, acc_means, acc_vars, best_one, cand_size, learning_rate):
        tmp_acc_means = acc_means.copy()
        tmp_acc_vars = acc_vars.copy()
        acc_means = acc_means/float(cand_size)
        acc_vars = acc_vars/float(cand_size)
        if None != best_one.params:
            means = means*(1.0 - learning_rate) + \
                                learning_rate*((acc_means + best_one.params) / 2.0)
        else:
            means = means*(1.0 - learning_rate) + \
                                learning_rate * (acc_means)

        vars = vars*(1-learning_rate) + (acc_vars - acc_means*acc_means)*learning_rate            
        return means, vars