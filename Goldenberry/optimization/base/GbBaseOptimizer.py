import abc

class GbBaseOptimizer(object):
    __metaclass__ = abc.ABCMeta
    """Base class for optimization algorithms."""
    
    cost_func = None
    name = ""
    
    @abc.abstractmethod
    def setup(self, **kwargs):
        """Setup the optimizer algorithm."""
        raise NotImplementedError()

    @abc.abstractmethod
    def ready(self):
        """Informs if the algorithm is ready to execute."""
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self):
        """Reset the current optimizer settings to be able to start again the search."""
        raise NotImplementedError

    @abc.abstractmethod
    def search(self):
        """Search for a optimal solution"""
        raise NotImplementedError()