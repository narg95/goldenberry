import numpy as np
import itertools as itr
from Goldenberry.statistics.distributions import BivariateBinomial, BinomialContingencyTable
from Goldenberry.optimization.edas.BaseEda import BaseEda
from Goldenberry.optimization.base.GbSolution import *
import math

class Bmda(BaseEda):
    """Bivariate marginal distribution algorithm."""

    cand_size = None
    vars_size = None
    cost_function = None
    distr = None
    max_iters = None
    iters = None
    percentile = None

    def setup(self, cost_function, var_size, cand_size, max_iters = None, percentile = 50):
        """Configure a Cga instance"""
        self.cand_size = cand_size
        self.vars_size = var_size
        self.cost_function = cost_function
        self.distr = BivariateBinomial(var_size)
        self.max_iters = max_iters
        self.iters = 0
        self.percentile = percentile
        
        # Graph properties
        self.marginals = None
        self.children = None
        self.cond_props = None

    def result_distribution(self):
        """Provides the final estimated distribution."""
        return self.distr

    def ready(self):
        """Informs if the algorithm is ready to execute."""
        return (None != self.cand_size \
                and None != self.vars_size\
                and None != self.cost_function \
                and None != self.distr \
                and self.cand_size > 0 \
                and self.vars_size > 0 
                and self.percentile > 0 \
                and self.percentile < 100 \
                and self.percentile > 0)

    def update_candidates(self, best):
            return np.concatenate((best, self.distr.sample(self.cand_size - best.shape[0])))
        
    def search(self):
        """Search for an optimal solution."""
        pop = self.distr.sample(self.cand_size)
        while not self.hasFinished():
            self.iters += 1            
            best = self.best_population(pop)
            self.update_distribution(best)
            pop = self.update_candidates(best)
        
        #returns the winner with its estimated cost
        winner = self.distr.sample(1)
        return GbSolution(winner, self.cost_function(winner))

    def best_population(self, pop):
        fit = self.cost_function(pop)
        index = np.argsort(fit)[:(self.cand_size * self.percentile/100):-1]
        return pop[index]
        
    def update_distribution(self, pop):
        self.marginals = np.average(pop, axis = 0)
        self.roots, self.children, self.cond_props = Bmda.generate_graph(pop)        
        self.distr = BivariateBinomial(p = self.marginals, cond_props = self.cond_props, children = self.children, roots = self.roots)

    @staticmethod
    def generate_graph(pop):
        
        var_size = pop.shape[1]
        
        #Check if there are vertices.
        if var_size < 1:
            raise Exception("No vertex to generate the graph.")

        #initialize variables
        children = [[] for i in xrange(var_size)]
        roots = []
        cond_props = [[] for i in xrange(var_size)]
                
        A = range(len(cond_props))
        A_ = []
        randidx = np.random.randint(0, len(A))
        v = A[randidx]
        roots.append(v)
        cond_props[v] = []
        A_.append(v)
        del A[randidx]

        # calculate chi_matrix
        chi_matrix = Bmda.calculate_chisquare_matrix(pop)
         
        while len(A) > 0:
            v1, v2, chi = Bmda.max_chisquare(A, A_, chi_matrix)
            if 0.0 != chi:
                children[v2].append(v1)
                ctable = BinomialContingencyTable(pop[:,[v2]], pop[:, [v1]])
                cond_props[v1] = ctable.PyGx[:, 1]
                if cond_props[v1] == []:
                    print "error"
                A_.append(v1)
                A.remove(v1)
            else:
                randidx = np.random.randint(0, len(A))
                v = A[randidx]
                roots.append(v)
                cond_props[v] = []
                A_.append(v)
                del A[randidx]
        
        return roots, children, cond_props

    def hasFinished(self):
        finish = not (self.max_iters is None) and self.iters > self.max_iters
        if finish:
            return True
        return (((1 - self.distr.p) < 0.01) | (self.distr.p < 0.01)).all()

    @property
    def distribution(self):
        return self.distr

    @staticmethod
    def calculate_chisquare_matrix(pop):
        _, length = pop.shape
        chi_matrix = np.empty((length, length))
        for x,y in itr.combinations(xrange(length), 2):
            ctable = BinomialContingencyTable(pop[:,[x]], pop[:, [y]])
            chisquare = ctable.chisquare()
            
            #independency threshold
            if math.isinf(chisquare):
                print "inf"
            if chisquare > 3.84:
                chi_matrix[x,y] = chi_matrix[y,x] = chisquare
            else:
                chi_matrix[x,y] = chi_matrix[y,x] = 0.0                
        return chi_matrix


    @staticmethod
    def max_chisquare(X, Y, chi_matrix):
        indx = np.array([(x,y) for x,y in itr.product(X, Y)])
        rows, cols = indx[:, 0], indx[:,1]
        maxidx = np.argmax(chi_matrix[rows, cols])
        (x,y) = indx[maxidx]
        max_chi = chi_matrix[x,y] 
        return x, y, max_chi
