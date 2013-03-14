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

    def generate_candidates(self, sample_size, best):
        if best == None:
            return GbBaseEda.generate_candidates(self, sample_size, best)
        
        candidates = GbBaseEda.generate_candidates(self,sample_size - len(best), best)
        return np.concatenate((best, candidates))[np.random.permutation(self.cand_size)]

    def update_distribution(self, best, best_one):
        marginals = np.average(best, axis = 0)
        roots, children, cond_props = Bmda.generate_graph(best)        
        self.distr = BivariateBinomial(p = marginals, cond_props = cond_props, children = children, roots = roots)

    @staticmethod
    def generate_graph(candidates):
        
        var_size = candidates.shape[1]
        
        # calculate chi_matrix
        chi_matrix = Bmda.calculate_chisquare_matrix(candidates)
        
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
                ctable = BinomialContingencyTable(candidates[:,[v2]], candidates[:, [v1]])
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

    @staticmethod
    def calculate_chisquare_matrix(candidates):
        _, length = candidates.shape
        chi_matrix = np.empty((length, length))
        for x,y in itr.combinations(xrange(length), 2):
            ctable = BinomialContingencyTable(candidates[:,[x]], candidates[:, [y]])
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
