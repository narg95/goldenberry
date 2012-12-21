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
            break



        #TODO: Review the NetworkX framework

    def hasFinished(self):
        finish = not (self.max_iters is None) and self.iters > self.max_iters
        if finish:
            return True
        return (((1 - self.distribution()) < 0.01) | (self.distribution() < 0.01)).all()

    def chisquare_all(self, i, V, pop):
        parent = pop[:, i]
        children = pop[:, V]
        px = np.average(parent, axis = 0)
        pys = np.average(children, axis = 0)
        pxys = BinomialContingencyTable(parent, children)
        for j in V:
            child = pop[:, j]
            py = np.average(child, axis = 0)
            cont = self.contingency(pop[:,x], pop[:,y])
            chisquare(parent, child, px, py, pxy)

    @classmethod
    def chisquare(X, Y, px, py, pxy):
        N = X.shape[0]
        px_y = px*py
        pxy_x_y = pxy - px_y
        return N*pxy_x_y.dot(pxy_x_y.T) / px_y

class BinomialContingencyTable:
    
    def __new__(self, X, Y):
        self._table = np.zeros(2, Y.shape[1] * 2)
        self._table[X, Y] += 1
        self._px = np.average(X, axis = 0)
        self._pys = np.average(X, axis = 0)
           
    @property
    def table(self):
        return self._table