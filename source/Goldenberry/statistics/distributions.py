import numpy as np
import itertools as it
import abc
import math
from collections import deque

class BaseDistribution:
    """Represents a base clase for probability distributions."""
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractproperty
    def parameters(self):
        """Gets the distribution parameters"""
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, **kwargs):
        """Samples based on the parameters in the actual distribution."""
        raise NotImplementedError

class Binomial(BaseDistribution):
    """Represents a binomial distribution."""
    
    def __init__(self, n = None, p = None):
        """Initialize a new binomial distribution."""
        if(None != n):
            self._n = n
            self._p =  np.tile(0.5,(1, n))
        elif(None != p) :
            self._n = p.size
            self._p = p
        else:
            raise ValueError("provide the variables size \
                or the parameters for initialize the binomial distribution")
    
    def __getitem__(self, key):
        return self._p[key]

    def __setitem__(self, key, value):
        self._p[key] = value

    def __call__(self):
        return self._p

    @property
    def parameters(self):
        return self._n, self._p

    def sample(self, sample_size):
        """Samples based on the current binomial parameters (variables size and bernoulli parameters)."""
        return np.matrix(np.random.rand(sample_size, self._n) <= np.ones((sample_size, 1)) * self._p, dtype=float)

class BivariateBinomial(BaseDistribution):
    
    def __init__(self, n = None, p = None, pyGx = None, edges = None, roots = None):
        if None != n:
            self.n = n
            self.p =  np.tile(0.5,(1, n))
            self.pyGx = np.tile(0.5,(2, n))
            self.edges = []
            self.roots = range(n)
            self.vertex = self.roots
        elif None != p and None != pyGx and None != edges:
            if pyGx.shape[1] != len(edges):
                raise AttributeError("Join probability must be the same size than the number of edges")
            self.n = p.shape[1]
            self.p = p
            self.pyGx = pyGx
            self.edges = edges
            self.vertex = range(self.n)
            if None == roots:
                self.roots = [x for x in self.vertex if np.all([x != c for _, c in edges])]
            else: 
                self.roots = roots
            
        else:
            raise AttributeError("Not enough parameters to create a Bivariate binomial distribution")
    
    @property
    def parameters(self):
        return self.n, self.p, self.pyGx, self.edges

    def sample(self, sample_size):       
        
        #TODO:  Avoid transfromation by receiving this as a parameter in the constructor
        D = [[] for i in xrange(self.n)]
        C = [[] for i in xrange(self.n)]
        Q = self.roots
        for idx, (ixp, ixc) in enumerate(self.edges):
            D[ixp].append(ixc)
            C[ixc] = self.pyGx[:, idx]

        
        """Samples based on the current bivariate binomial parameters ."""
        # Bug in numpy with dtype = int and indexing arrays [].
        samples = np.zeros((sample_size, self.n), dtype=int)
        
        # samples univiariate probabilities
        samples[:, self.roots] = np.random.rand(sample_size, len(self.roots)) <= np.ones((sample_size, 1)) * self.p[:, self.roots]

        while len(Q) > 0:
            parent = Q.pop()
            for chl in D[parent]:
                cond_probs = C[chl][samples[:, parent]]
                samples[:, chl] = (np.random.rand(sample_size) <= cond_probs)
                Q.append(chl)

        return samples

class BinomialContingencyTable:
    
    def __init__(self, X, Y):
        self._n, self.l = Y.shape
        self._table = np.zeros((2, self.L * 2), dtype=int)
        yt = np.array(range(0, self.L*2, 2), dtype=int)
        for i in xrange(self.N):
            self._table[X[i], yt + Y[i]] += 1
        self._px = np.average(X, axis = 0)
        self._pys = np.average(Y, axis = 0)
        self._pxys = self._table / float(self._n);
           
    @property
    def Table(self):
        return self._table

    @property
    def N(self):
        return self._n

    @property
    def L(self):
        return self.l

    @property
    def Pxys(self):
        return self._pxys

    @property
    def Px(self):
        return self._px

    @property
    def Pys(self):
        return self._pys

    def chisquare(self):
        chi = np.zeros(self.L)
        for i in xrange(self.L):
            pxy = self.Pxys[:, [2*i, 2*i+1]]
            px_py = np.array([1- self.Px, self.Px]) * np.array([1 - self.Pys[i], self.Pys[i]])
            val = (pxy - px_py)
            val = np.array([0 if  math.isnan(j) else j for j in ((val * val)/px_py).flat])
            chi[i] = np.sum(val)

        return self.N*chi

def _splitter(data, pred):
    yes, no = [], []
    for d in data:
        (yes if pred(d) else no).append(d)
    return [yes, no]
    
    
