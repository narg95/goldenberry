import numpy as np

class BaseDistribution(object):
    """Represents a base clase for probability distributions."""
    
    @property
    def parameters(self):
        pass

class Binomial(BaseDistribution):
    """Represents a binomial distribution."""
    
    def __init__(self, vars_size = None, parameters = None):
        """Initialize a new binomial distribution."""
        if(None != vars_size):
            self._parameters = np.zeros(var_size)
        elif(None != parameters) :
            self._parameters = parameters
        else:
            raise ValueError("provide the variables size \
                or the parameters for initialize the binomial distribution")

    def __call__(self):
        return self._parameters
    
    def __getitem__(self, key):
        return self._parameters[key]

    @property
    def parameters(self):
        return self._parameters