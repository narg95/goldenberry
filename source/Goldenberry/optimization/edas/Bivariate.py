import numpy as np
import itertools as itr
from Goldenberry.statistics.distributions import BivariateBinomial, BinomialContingencyTable
from Goldenberry.optimization.edas.GbBaseEda import GbBaseEda
from Goldenberry.optimization.base.GbSolution import GbSolution
import math

class Bmda(GbBaseEda):
    """Bivariate marginal distribution algorithm."""

    def initialize(self):
        self.distr = BivariateBinomial( n = self.var_size)

    def sample(self, sample_size, top_ranked, best):
        if top_ranked is None:
            return GbBaseEda.sample(self, sample_size, top_ranked, best)
        
        candidates = GbBaseEda.sample(self,sample_size - len(top_ranked), top_ranked, best)
        return np.concatenate((top_ranked, candidates))[np.random.permutation(self.cand_size)]

    def estimate(self, top_ranked, best):
        marginals = np.average(top_ranked, axis = 0)
        entropy =  - marginals * np.log2(marginals) - (1 - marginals) * np.log2(1 - marginals)
        roots, children, cond_props = Bmda.build_graph(top_ranked, entropy)        
        self.distr = BivariateBinomial(p = marginals, cond_props = cond_props, children = children, roots = roots)

    @staticmethod
    def build_graph(candidates, sort):
        
        var_size = candidates.shape[1]
        
        # calculate chi_matrix
        chi_matrix = Bmda._get_chisqr_matrix(candidates)
        
        #Check if there are vertices.
        if var_size < 1:
            raise Exception("No vertex to generate the graph.")

        #initialize variables
        children = [[] for i in xrange(var_size)]
        roots = []
        cond_props = [[] for i in xrange(var_size)]
                
        A = range(len(cond_props))
        A_ = []
        idx = np.argmin(sort)
        v = A[idx]
        roots.append(v)
        cond_props[v] = []
        del A[idx]
        A_.append(v)        

        while len(A) > 0:
            v1, v2, chi = Bmda._max_chisqr(A, A_, chi_matrix)
            if chi > 3.84:
                children[v2].append(v1)
                ctable = BinomialContingencyTable(candidates[:,[v2]], candidates[:, [v1]])
                cond_props[v1] = ctable.PyGx[:, 1]                
                A_.append(v1)
                A.remove(v1)
            else:
                idx = np.argmin(sort[A])
                v = A[idx]
                roots.append(v)
                cond_props[v] = []
                #TODO:  Review if a memory problem is caused for reinitializing the list.
                A_ = [v]
                del A[idx]
        
        return roots, children, cond_props

    @staticmethod
    def _get_chisqr_matrix(candidates):
        _, length = candidates.shape
        chi_matrix = np.empty((length, length))
        for x,y in itr.combinations(xrange(length), 2):
            ctable = BinomialContingencyTable(candidates[:,[x]], candidates[:, [y]])
            chisquare = ctable.chisquare()
            chi_matrix[x,y] = chi_matrix[y,x] = chisquare            
        return chi_matrix

    @staticmethod
    def _max_chisqr(X, Y, chi_matrix):
        indx = np.array([(x,y) for x,y in itr.product(X, Y)])
        rows, cols = indx[:, 0], indx[:,1]
        maxidx = np.argmax(chi_matrix[rows, cols])
        (x,y) = indx[maxidx]
        max_chi = chi_matrix[x,y] 
        return x, y, max_chi
