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
        raise NotImplementedError()

    @abc.abstractmethod
    def sample(self, **kwargs):
        """Samples based on the parameters in the actual distribution."""
        raise NotImplementedError()

    @abc.abstractmethod
    def has_converged(self):
        """"Informs wheter or not the distribution has converged."""
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self):
        """Resets the current distribution."""
        raise NotImplementedError()

class Binomial(BaseDistribution):
    """Represents a binomial distribution."""
    
    def __init__(self, n = None, p = None):
        """Initialize a new binomial distribution."""
        if(None != n):
            self.n = n
            self.reset()
        elif(None != p) :
            self.n = p.size
            self.p = p
        else:
            raise ValueError("provide the variables size \
                or the parameters for initialize the binomial distribution")
    
    def __getitem__(self, key):
        return self.p[key]

    def __setitem__(self, key, value):
        self.p[key] = value

    def __call__(self):
        return self.p

    @property
    def parameters(self):
        return self.n, self.p

    def sample(self, sample_size):
        """Samples based on the current binomial parameters (variables size and bernoulli parameters)."""
        return np.array(np.random.rand(sample_size, self.n) <= np.ones((sample_size, 1)) * self.p, dtype=float)

    def has_converged(self):
        return (((1 - self.p) < 0.01) | (self.p < 0.01)).all()

    def reset(self):
        self.p =  np.tile(0.5,(1, self.n))

class BivariateBinomial(BaseDistribution):
    
    def __init__(self, n = None, p = None, cond_props = None, children = None, roots = None):
        if None != n:
            self.n = n
            self.reset()
        elif None != p and None != cond_props and None != children:
            if len(cond_props) != len(children):
                raise AttributeError("Join probability must be the same size than the number of edges")
            self.n = p.size
            self.p = p
            self.cond_props = cond_props
            self.children = children
            self.vertex = range(self.n)
            if None == roots:
                self.roots = [idx for idx, x in enumerate(self.cond_props) if x == []]
            else: 
                self.roots = roots
            
        else:
            raise AttributeError("Not enough parameters to create a Bivariate binomial distribution")
    
    def reset(self):
        self.p =  np.tile(0.5,(1, self.n))
        self.children = [[] for i in xrange(self.n)]
        self.cond_props = [[] for i in xrange(self.n)]
        self.roots = range(self.n)
        self.vertex = range(self.n)

    def has_converged(self):
        return (((1 - self.p) < 0.01) | (self.p < 0.01)).all()

    @property
    def parameters(self):
        return self.n, self.p, self.cond_props, self.children

    def sample(self, sample_size):       
        
        # initialize queue
        Q = [i for i in self.roots]
        
        """Samples based on the current bivariate binomial parameters ."""
        # Bug in numpy with dtype = int and indexing arrays [].
        samples = np.zeros((sample_size, self.n), dtype=int)
        
        # samples univiariate probabilities
        samples[:, self.roots] = np.random.rand(sample_size, len(self.roots)) <= np.ones((sample_size, 1)) * self.p[:, self.roots]

        while len(Q) > 0:
            parent = Q.pop()
            for chl in self.children[parent]:
                if self.cond_props[chl] == []:
                    raise Exception("the given children has not a parent nor a conditional probability")
                pyGx = self.cond_props[chl][samples[:, parent]]
                samples[:, chl] = np.array(np.random.rand(sample_size) <= pyGx)
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
    
    @property
    def PyGx(self):
         p = self.pxys / np.array([[1 - self.px, 1 - self.px],[self.px, self.px]], dtype = float)
         p[~np.isfinite(p)] = 0.0
         return p
              
    def chisquare(self):
        chi = np.zeros(self.l)
        for i in xrange(self.l):
            pxy = self.pxys[:, [2*i, 2*i+1]]
            px_py = np.array([[1- self.px],[self.px]]).dot(np.array([[1 - self.pys[i]], [self.pys[i]]]).T)
            val = (pxy - px_py)
            val = np.array([0 if  np.isnan(j) or np.isinf(j) else j for j in ((val * val)/px_py).flat])
            chi[i] = np.sum(val)

        return self.n*chi

def _splitter(data, pred):
    yes, no = [], []
    for d in data:
        (yes if pred(d) else no).append(d)
    return [yes, no]
    
    
