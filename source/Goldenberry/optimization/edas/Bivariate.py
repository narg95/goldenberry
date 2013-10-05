import numpy as np
import itertools as itr
from Goldenberry.statistics.distributions import BivariateBinomial, BinomialContingencyTable
from Goldenberry.optimization.edas.GbBaseEda import GbBaseEda
from Goldenberry.optimization.base.GbSolution import GbSolution
import math

class DependencyMethod:
    """ This is an enumeration for the supported dependency methods"""
    chi2_test = 0,
    sim = 1

class Bmda(GbBaseEda):
    """Bivariate marginal distribution algorithm."""

    def setup(self, cand_size = 20, max_evals = 100, selection_rate = 50, learning_rate = 1, callback_func = None, dependency_method = DependencyMethod.chi2_test, **kwargs):
        return super(Bmda, self).setup(cand_size, max_evals, selection_rate, learning_rate, callback_func, dependency_method = DependencyMethod.chi2_test, **kwargs)
    
    def initialize(self):
        self.distr = BivariateBinomial( n = self.var_size)
        
    def sample(self, sample_size, top_ranked, best):
        if top_ranked is None:
            return GbBaseEda.sample(self, sample_size, top_ranked, best)
        
        candidates = GbBaseEda.sample(self,sample_size - len(top_ranked), top_ranked, best)
        return np.concatenate((top_ranked, candidates))[np.random.permutation(self.cand_size)]

    def build_solution(self, params, cost):
        return GbSolution(params, cost, self.distr.roots, self.distr.children)

    def estimate(self, top_ranked, best):
        marginals = np.average(top_ranked, axis = 0)
        entropy =  - marginals * np.log2(marginals) - (1 - marginals) * np.log2(1 - marginals)
        roots, children, cond_props = Bmda.build_graph(top_ranked, entropy, self.dependency_method)        
        self.distr = BivariateBinomial(p = marginals, cond_props = cond_props, children = children, roots = roots)

    @staticmethod
    def build_graph(candidates, sort, method):
        
        var_size = candidates.shape[1]
        
        # calculate dependency matrix
        dependency_matrix = Bmda.build_chi2_dependency_matrix(candidates) if method == DependencyMethod.chi2_test else Bmda.build_sim_dependency_matrix(candidates)
        
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
            v1, v2, chi = Bmda.get_max_dependency(A, A_, dependency_matrix)
            if chi > 0.0:
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
    def build_chi2_dependency_matrix(candidates):
        _, length = candidates.shape
        chi_matrix = np.empty((length, length))
        for x,y in itr.combinations(xrange(length), 2):
            ctable = BinomialContingencyTable(candidates[:,[x]], candidates[:, [y]])
            chisquare = ctable.chisquare()
            chi_matrix[x,y] = chi_matrix[y,x] = chisquare if chisquare >= 3.84 else 0.0
        return chi_matrix

    @staticmethod
    def build_sim_dependency_matrix(candidates):
        _, length = candidates.shape
        sim_matrix = np.empty((length, length))
        for x,y in itr.combinations(xrange(length), 2):
            ctable = BinomialContingencyTable(candidates[:,[x]], candidates[:, [y]])
            sim =  ctable.sim()
            sim_matrix[x,y] = sim_matrix[y,x] = sim if sim > 1e-5 else 0.0
        return sim_matrix

    @staticmethod
    def get_max_dependency(X, Y, dependency_matrix):
        indx = np.array([(x,y) for x,y in itr.product(X, Y)])
        rows, cols = indx[:, 0], indx[:,1]
        maxidx = np.argmax(dependency_matrix[rows, cols])
        (x,y) = indx[maxidx]
        max_chi = dependency_matrix[x,y] 
        return x, y, max_chi
