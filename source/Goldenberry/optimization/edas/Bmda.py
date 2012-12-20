import numpy as np
from Goldenberry.optimization.edas.distributions import *
from Goldenberry.optimization.edas.BaseEda import BaseEda
from Goldenberry.optimization.base.GbSolution import *

class Bmda(BaseEda):
    """Bivariate marginal distribution algorithm."""

    pop_size = None
    vars_size = None
    cost_function = None
    distribution = None
    max_iters = None
    iters = None
    percentile = None

    def setup(self, cost_function, varsize, popsize, maxiters = None, percentile = 50):
        """Configure a Cga instance"""
        self._pop_size = popsize
        self._vars_size = varsize
        self._cost_function = cost_function
        self._distribution = Binomial(params = np.tile(0.5,(1,varsize)))
        self._max_iters = maxiters
        self._iters = 0
        self.percentile = percentile
        self.edges

    def result_distribution(self):
        """Provides the final estimated distribution."""

        pass 

    def ready(self):
        """Informs if the algorithm is ready to execute."""
        return (None != self.pop_size \
                and None != self.vars_size\
                and None != self.cost_function \
                and None != self.distribution \
                and self.pop_size > 0 \
                and self.vars_size > 0 
                and self.percentile > 0 \
                and self.percentile < 100)

    def search(self):
        """Search for an optimal solution."""
        while not self.hasFinished():
            self.iters += 1
            pop = self.distribution.sample(_pop_size)
            best = self.best_population(pop)
            self.estimate_distribution(best)
        
        #returns the winner with its estimated cost
        winner = self.distribution.sample(1)
        return GbSolution(winner, self.cost_function(winner))

    def best_population(self, pop):
        fit = self.cost_function(pop)
        minvalue = np.percentile(fit, self.percentile)
        return pop[np.where(fit > minvalue)]
        
    def estimate_distribution(self, pop):
        graph = self.generate_graph(pop, px)

    def generate_graph(self, pop):
        px = np.average(a, axis = 0)
        
        V = range(self.vars_size)
        A = range(self.vars_size)
        E = []
        R = []
        edges = []

        # We assume A is >= 1
        v = A[0]
        R.append(v)
        del A[0]

        while len(A) > 0:
            



        #TODO: Review the NetworkX framework

    def hasFinished(self):
        finish = not (self.max_iters is None) and self.iters > self.max_iters
        if finish:
            return True
        return (((1 - self.distribution()) < 0.01) | (self.distribution() < 0.01)).all()

    def chi_square(self, i, V, pop):
        parent = pop[:, i]
        pxy = np.zeros(2, len(V) * 2)
        for j in V:
            child = pop[:, j]


        