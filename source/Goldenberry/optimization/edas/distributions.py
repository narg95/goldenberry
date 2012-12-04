import numpy as np
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
        elif(None != params) :
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
        return self._n

    @property
    def parameters(self):
        return self._n, self._p

    def sample(self, sample_size):
        """Samples based on the current binomial parameters (variables size and bernoulli parameters)."""
        return np.matrix(np.random.rand(sample_size, self._n) <= np.ones((sample_size, 1)) * self._p, dtype=float)

class BivariateBinomial(BaseDistribution):
    
    def __init__(self, n, p, pxy):
        if None != n:
            self.n = n
            self.p =  np.tile(0.5,(1, n))
            self.pxy = np.tile(0.5,(2, n))
            self.edges = {}
            self.roots = [[range(n)]]
            self.vertex = [[range(n)]]
   
    @property
    def parameters(self):
        return self.n, self._px, self._py, self.pxy, self.edges

    def sample(self, sample_size):
        """Samples based on the current bivariate binomial parameters ."""
        samples = np.zeros((sample_size, n))
        
        # samples univiariate probabilities
        samples[:, self.roots] = np.matrix(np.random.rand(sample_size, len(self.roots)) <= np.ones((sample_size, 1)) * self.p[:, self.roots], dtype=float)

        # samples conditional dependencies
        ipars, ichln = splitter(self.vertex, lambda item: item in self.roots)
        while ichln.count() > 0:
            for iedg, edg in enumerate(self.edges):
                for ip in ipars:
                    for ic in ichln:
                        if edg == (self.vertex[ip], self.vertex[ic]):
                            vals = samples[:, ip]
                            self.join[vals, iedg]/abs((vals - 1) + self.p) 

        return np.matrix(np.random.rand(sample_size, self.vars_size) <= np.ones((sample_size, 1)) * self.p, dtype=float)  


    def isplitter(data, pred):
        yes, no = [], []
        for d in data:
            (yes if pred(d) else no).append(i)
        return [yes, no]
    
    

