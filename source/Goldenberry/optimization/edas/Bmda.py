import numpy as np
import itertools as itr
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
        self._distribution = BivariateBinomial(varsize)
        self._max_iters = maxiters
        self._iters = 0
        self.percentile = percentile
        
        #TODO: Review the NetworkX framework
        self.E = []
        self.D = {}
        self.R = []
        self.V = range(self.vars_size)
        self.PyGx = []
        self.P = []

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
        self.generate_graph(pop, px)
        self.distribution = BivariateBinomial(p = p, pyGx = self.PyGx, edges = self.E)
        samples = dist.sample(20)

    def generate_graph(self, pop):
        px = np.average(a, axis = 0)
        
        #initialize local variables
        A = range(self.vars_size)        
        A_ = []
       
        # We assume A is >= 1
        #TODO: we take always A[0] but a random one must be used
        v = A[0]
        R.append(v)
        A_.append(v)
        del A[0]
         
        while len(A) > 0:
            v1, v2, chi, ctable = Bmda.get_max_chisquare(A, A_, pop)
            if None != chi:
                self.E.append((v1, v2))
                self.PyGx.append(ctable.pyGx)
                self.P[v1] = ctable.
                A_.append(v1)
                A.remove(v1)
            else:
                #TODO: we take always A[0] but a random one must be used
                v = A[0]
                R.append(v)
                A_.append(v)
                del A[0]            

    def hasFinished(self):
        finish = not (self.max_iters is None) and self.iters > self.max_iters
        if finish:
            return True
        return (((1 - self.distribution()) < 0.01) | (self.distribution() < 0.01)).all()

    def get_max_chisquare(self, R, A, pop):
        max_chi = 0.0
        max_a = max_b = None
        max_ctable = None
        for a, b in Bmda.product(R, A):
            chi, table = self.D.get((a,b), d = (None, None))
            if None == chi:
                ctable = BinomialContingencyTable(pop[a], pop[b])
                chi = ctable.chisquare()
                self.D[(a,b)] = chi, ctable
            if chi >= 3.84 and chi > max_chi:
                max_chi = chi
                max_a = a
                max_b = b
                max_ctable = ctable
        return max_a, max_b, None, None if max_a == None else max_a, max_b, max_chi, max_ctable

    @staticmethod
    def __product__(A, B):
        for a in A:
            for b in B:
                yield a,b if b > a else b, a        