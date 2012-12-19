import numpy as np
import itertools as it
import abc

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
            self.roots = [e for e in range(n)]
            self.vertex = range(n)
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
        """Samples based on the current bivariate binomial parameters ."""
        # Bug in numpy with dtype = int and indexing arrays [].
        samples = np.zeros((sample_size, self.n), dtype=int)
        
        # samples univiariate probabilities
        samples[:, self.roots] = np.random.rand(sample_size, len(self.roots)) <= np.ones((sample_size, 1)) * self.p[:, self.roots]

        # samples conditional dependencies
        ipars, ichln = _splitter(self.vertex, lambda item: item in self.roots)
        while len(ichln) > 0:
            ic  = ichln[0]
            for idx, (iep, iec)  in enumerate(self.edges):
                # if child is in the current edge
                if iec == ic:
                    # if parent from current edge has been already processed.
                    if iep in ipars:
                        cond_probs = self.pyGx[samples[:, iep], idx]
                        samples[:, ic] = (np.random.rand(sample_size) <= cond_probs)
                        ipars.append(ic)
                        del ichln[0]
                    break
        return samples
    
def _splitter(data, pred):
    yes, no = [], []
    for d in data:
        (yes if pred(d) else no).append(d)
    return [yes, no]
    
    

