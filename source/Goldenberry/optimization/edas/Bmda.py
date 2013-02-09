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
    percentage = None
    fit_evals = None
    max_evals = None

    def setup(self, cost_function, var_size, cand_size, percentage = 50, max_iters = None, max_evals = None):
        """Configure a Bmda instance"""
        self.cand_size = cand_size
        self.vars_size = var_size
        self.cost_function = cost_function
        self.distr = BivariateBinomial(var_size)
        self.max_iters = max_iters
        self.iters = 0
        self.fit_evals = 0
        self.percentage = percentage
        self.max_evals = max_evals
        self.evals_per_gens = cand_size * (100 - percentage)/100
        
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
                and self.percentage > 0 \
                and self.percentage < 100 \
                and self.percentage > 0)

    def search(self):
        """Search for an optimal solution."""
        best_candidate = GbSolution(None, 0.0)
        pop = self.distr.sample(self.cand_size)
        while not self.hasFinished():
            
            bests, winner = self.best_candidates(pop)
            self.update_distribution(bests)
            pop = self.update_candidates(bests)
            
            if best_candidate.cost < winner.cost:
                best_candidate = winner

            self.iters += 1
            self.fit_evals += self.evals_per_gens

        #returns the winner with its estimated cost
        return best_candidate

    def update_candidates(self, best):
            return np.concatenate((best, self.distr.sample(self.cand_size - best.shape[0])))[np.random.permutation(self.cand_size)]

    def best_candidates(self, bests):
        fits = self.cost_function(bests)
        index = np.argsort(fits)[:(self.cand_size * self.percentage/100):-1]
        return bests[index], GbSolution(bests[index[0]], fits[index[0]])
        
    def update_distribution(self, pop):
        self.marginals = np.average(pop, axis = 0)
        self.roots, self.children, self.cond_props = Bmda.generate_graph(pop)        
        self.distr = BivariateBinomial(p = self.marginals, cond_props = self.cond_props, children = self.children, roots = self.roots)

    @staticmethod
    def generate_graph(pop):
        
        var_size = pop.shape[1]
        
        # calculate chi_matrix
        chi_matrix = Bmda.calculate_chisquare_matrix(pop)
        
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
        del A[randidx]
        A_.append(v)        

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
