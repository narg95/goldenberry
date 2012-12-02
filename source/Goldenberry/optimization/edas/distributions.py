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
        return np.matrix(np.random.rand(sample_size, self.vars_size) <= np.ones((sample_size, 1)) * self._p, dtype=float)

class BivariateBinomial(BaseDistribution):
    
    def __init__(self, n, px, py, pxy):
        if None != n:
            self._n = n
            self._px =  np.tile(0.5,(1, n))
            self._py = np.tile(0.5,(2, n))
            self._pxy = 

   
    @property
    def parameters(self):
        return self._n, self._p

    def sample(self, sample_size):
        """Samples based on the current binomial parameters (variables size and bernoulli parameters)."""
        return np.matrix(np.random.rand(sample_size, self.vars_size) <= np.ones((sample_size, 1)) * self._p, dtype=float)  



    
    

