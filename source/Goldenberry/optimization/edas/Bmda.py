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
        self._distribution = Binomial(params = np.tile(0.5,(1,varsize)))
        self._max_iters = maxiters
        self._iters = 0
        self.percentile = percentile
        
        #TODO: Review the NetworkX framework
        self.E = []
        self.D = {}
        self.R = []
        self.V = range(self.vars_size)

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
            v1, v2, chi = Bmda.get_max_chisquare(A, A_, pop)
            if None != chi:
                self.E.append((v1, v2))
                A_.append(v1)
                A.remove(v1)
            else:
                #TODO: we take always A[0] but a random one must be used
                v = A[0]
                R.append(v)
                A_.append(v)
                del A[0]
        
        #TODO: Estimate starting from R as the roots.
        

    def hasFinished(self):
        finish = not (self.max_iters is None) and self.iters > self.max_iters
        if finish:
            return True
        return (((1 - self.distribution()) < 0.01) | (self.distribution() < 0.01)).all()

    @staticmethod
    def chisquare(i, V, pop):
        parent = pop[:, i]
        children = pop[:, V]
        N = ctable.N
        pys = np.array([ctable.Pys[i/2] if i%2 == 1 else 1 - ctable.Pys[i/2] for i in xrange(ctable.L * 2)])
        px = np.array([[1- ctable.Px],[ctable.Px]])
        A = px * pys
        B = ctable.Pxys - A
        return N*(B/A).dot(B.T) 

    def get_max_chisquare(self, R, A, pop):
        max_chi = 0.0
        max_a = max_b = None
        for a,b in Bmda.product(R, A):
            chi = self.D.get((a,b))
            if None == chi:
                chi = Bmda.chisquare(a, b, pop)
                self.D[(a,b)] = chi
            if chi >= 3.84 and chi > max_chi:
                max_chi = chi
                max_a = a
                max_b = b
        return max_a, max_b, None if max_a == None else max_chi

    @staticmethod
    def __product__(A, B):
        for a in A:
            for b in B:
                yield a,b if b > a else b, a        