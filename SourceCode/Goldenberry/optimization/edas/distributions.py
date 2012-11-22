import numpy as np
import abc

class BaseDistribution:
    """Represents a base clase for probability distributions."""
    __metaclass__ = abc.ABCMeta
    
    def __call__(self):
        return self.parameters
    
    @abc.abstractproperty
    def parameters(self):
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, **kwargs):
        raise NotImplementedError

    

class Binomial(BaseDistribution):
    """Represents a binomial distribution."""
    
    def __init__(self, vars_size = None, parameters = None):
        """Initialize a new binomial distribution."""
        if(None != vars_size):
            self._var_size = var_size
            self._parameters = np.zeros(var_size)
        elif(None != parameters) :
            self._var_size = parameters.size
            self._parameters = parameters
        else:
            raise ValueError("provide the variables size \
                or the parameters for initialize the binomial distribution")
    
    def __getitem__(self, key):
        return self._parameters[key]

    def __setitem__(self, key, value):
        self._parameters[key] = value

    @property
    def parameters(self):
        return self._parameters

    def sample(self, sample_size):
        """Samples based on the current binomial parameters (variables size and bernoulli parameters)."""
        return np.matrix(np.random.rand(sample_size, self._var_size) <= np.ones((sample_size, 1)) * self._parameters, dtype=float)

