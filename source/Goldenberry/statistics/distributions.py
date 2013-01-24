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

    @property
    def P(self):
        return self._p

    @P.setter
    def P(self, value):
        self._p = value

    def sample(self, sample_size):
        """Samples based on the current binomial parameters (variables size and bernoulli parameters)."""
        return np.array(np.random.rand(sample_size, self._n) <= np.ones((sample_size, 1)) * self._p, dtype=float)

class BivariateBinomial(BaseDistribution):
    
    def __init__(self, n = None, p = None, cond_props = None, children = None, roots = None):
        if None != n:
            self.n = n
            self.p =  np.tile(0.5,(1, n))
            self.children = [[] for i in xrange(n)]
            self.cond_props = [[] for i in xrange(n)]
            self.roots = range(n)
            self.vertex = self.roots
        elif None != p and None != cond_props and None != children:
            if cond_props.shape[1] != len(children):
                raise AttributeError("Join probability must be the same size than the number of edges")
            self.n = p.shape[1]
            self.p = p
            self.cond_props = cond_props
            self.children = children
            self.vertex = range(self.n)
            if None == roots:
                self.roots = [x for x in self.vertex if np.all([x != c for _, c in children])]
            else: 
                self.roots = roots
            
        else:
            raise AttributeError("Not enough parameters to create a Bivariate binomial distribution")
    
    @property
    def parameters(self):
        return self.n, self.p, self.cond_props, self.children

    def sample(self, sample_size):       
        
        Q = self.roots
        
        """Samples based on the current bivariate binomial parameters ."""
        # Bug in numpy with dtype = int and indexing arrays [].
        samples = np.zeros((sample_size, self.n), dtype=int)
        
        # samples univiariate probabilities
        samples[:, self.roots] = np.random.rand(sample_size, len(self.roots)) <= np.ones((sample_size, 1)) * self.p[:, self.roots]

        while len(Q) > 0:
            parent = Q.pop()
            for chl in children[parent]:
                cond_probs = cond_props[chl][samples[:, parent]]
                samples[:, chl] = (np.random.rand(sample_size) <= cond_probs)
                Q.append(chl)

        return samples

class BinomialContingencyTable:
    
    def __init__(self, X, Y):
        self.n, self.l = Y.shape
        self.table = np.zeros((2, self.l * 2), dtype=int)
        yt = np.array(range(0, self.l*2, 2), dtype=int)
        for i in xrange(self.n):
            self.table[X[i], yt + Y[i]] += 1
        self.px = np.average(X)
        self.pys = np.average(Y, axis = 0)
        self.pxys = self.table / float(self.n);
           
    def chisquare(self):
        chi = np.zeros(self.l)
        for i in xrange(self.l):
            pxy = self.pxys[:, [2*i, 2*i+1]]
            px_py = np.array([1- self.px, self.px]) * np.array([1 - self.pys[i], self.pys[i]])
            val = (pxy - px_py)
            val = np.array([0 if  math.isnan(j) else j for j in ((val * val)/px_py).flat])
            chi[i] = np.sum(val)

        return self.n*chi

def _splitter(data, pred):
    yes, no = [], []
    for d in data:
        (yes if pred(d) else no).append(d)
    return [yes, no]
    
    
