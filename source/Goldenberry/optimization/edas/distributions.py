import numpy as np
import abc

class BaseDistribution:
    """Represents a base clase for probability distributions."""
    __metaclass__ = abc.ABCMeta
    
    def __call__(self):
        return self.parameters
    
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
    
    _var_size = 0
    _params = []

    def __init__(self, vars_size = None, params = None):
        """Initialize a new binomial distribution."""
        if(None != vars_size):
            self._vars_size = vars_size
            self._params = np.zeros(vars_size)
        elif(None != params) :
            self._vars_size = params.size
            self._params = params
        else:
            raise ValueError("provide the variables size \
                or the parameters for initialize the binomial distribution")
    
    def __getitem__(self, key):
        return self._params[key]

    def __setitem__(self, key, value):
        self._params[key] = value

    @property
    def parameters(self):
        return self._params

    @property
    def vars_size(self):
        return self._vars_size

    def sample(self, sample_size):
        """Samples based on the current binomial parameters (variables size and bernoulli parameters)."""
        return np.matrix(np.random.rand(sample_size, self.vars_size) <= np.ones((sample_size, 1)) * self._params, dtype=float)

