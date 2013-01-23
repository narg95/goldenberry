import numpy as np
import itertools as itr
from Goldenberry.statistics.distributions import BivariateBinomial, BinomialContingencyTable
from Goldenberry.optimization.edas.BaseEda import BaseEda
from Goldenberry.optimization.base.GbSolution import *
import math

class Bmda(BaseEda):
    """Bivariate marginal distribution algorithm."""

    pop_size = None
    vars_size = None
    cost_function = None
    distr = None
    max_iters = None
    iters = None
    percentile = None

    def setup(self, cost_function, varsize, popsize, maxiters = None, percentile = 50):
        """Configure a Cga instance"""
        self.pop_size = popsize
        self.vars_size = varsize
        self.cost_function = cost_function
        self.distr = BivariateBinomial(varsize)
        self.max_iters = maxiters
        self.iters = 0
        self.percentile = percentile
        
        # Graph properties
        self.children = [[] for i in xrange(self.vars_size)]
        self.roots = []
        self.vertex = range(self.vars_size)
        self.cond_prop = [[] for i in xrange(self.vars_size)]
        self.marginals = []

    def result_distribution(self):
        """Provides the final estimated distribution."""
        return self.distr

    def ready(self):
        """Informs if the algorithm is ready to execute."""
        return (None != self.pop_size \
                and None != self.vars_size\
                and None != self.cost_function \
                and None != self.distr \
                and self.pop_size > 0 \
                and self.vars_size > 0 
                and self.percentile > 0 \
                and self.percentile < 100)

    def update_population(self, best):
            pop = np.concatenate(best, self.distr.sample(20))
        
    def search(self):
        """Search for an optimal solution."""
        pop = self.distr.sample(self.pop_size)
        while not self.hasFinished():
            self.iters += 1            
            best = self.best_population(pop)
            self.update_distribution(best)
            self.update_population(best)
        
        #returns the winner with its estimated cost
        winner = self.distr.sample(1)
        return GbSolution(winner, self.cost_function(winner))

    def best_population(self, pop):
        fit = self.cost_function(pop)
        minvalue = np.percentile(fit, self.percentile)
        return pop[np.where(fit > minvalue)]
        
    def update_distribution(self, pop):
        self.marginals = np.average(pop, axis = 0)
        Bmda.generate_graph(pop, self.roots, self.children)        
        self.distr = BivariateBinomial(p = self.marginals, pyGx = self.cond_prop, edges = self.children, roots = self.roots)

    @staticmethod
    def generate_graph(pop, roots, children):
        #initialize local variables
        A = range(self.vars_size)        
        A_ = []
        
        # We assume A is >= 1
        randidx = np.random.randint(0, len(A))
        v = A[randidx]
        roots.append(v)
        A_.append(v)
        del A[randidx]

        # calculate chi_matrix
        chi_matrix = Bmda.calculate_chisquare_matrix(pop)
         
        while len(A) > 0:
            v1, v2, chi = Bmda.max_chisquare(A, A_, pop)
            if 0.0 != chi:
                children[v1].append(v2)
                A_.append(v1)
                A.remove(v1)
            else:
                randidx = np.random.randint(0, len(A))
                v = A[randidx]
                roots.append(v)
                A_.append(v)
                del A[randidx]            

    def hasFinished(self):
        finish = not (self.max_iters is None) and self.iters > self.max_iters
        if finish:
            return True
        return (((1 - self.distr.p) < 0.01) | (self.distr.p < 0.01)).all()

    @property
    def distribution(self):
        return self.distr

    @staticmethod
    def calculate_chisquare_matrix(pop, cond_probs):
        _, length = pop.shape
        chi_matrix = np.empty((length, length))
        for x,y in itr.combinations(xrange(length), 2):
            ctable = BinomialContingencyTable(pop[:,[x]], pop[:, [y]])
            chisquare = ctable.chisquare()
            
            #independency threshold
            if chisquare > 3.84:
                chi_matrix[x,y] = chi_matrix[y,x] = chisquare
                cond_probs[y] = ctable.pxys
            else:
                chi_matrix[x,y] = chi_matrix[y,x] = 0.0
                cond_probs[y] = None
        return chi_matrix


    @staticmethod
    def max_chisquare(X, Y, pop, chi_matrix):
        maxidx = np.argmax(chi_matrix[itr.product(X, Y)])
        (x,y) = (X[maxidx % len(X)], Y[maxidx / len(Y)])
        max_chi = chi_matrix[x,y] 
        return x,y,max_chi
