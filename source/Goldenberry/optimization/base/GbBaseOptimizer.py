import abc

class GbBaseOptimizer(object):
    __metaclass__ = abc.ABCMeta
    """Base class for optimization algorithms."""

    @abc.abstractmethod
    def setup(self, cost_function, **kwargs):
        """Setup the optimizer algorithm."""
        raise NotImplementedError

    @abc.abstractmethod
    def search(self):
        """Search for a optimal solution"""
        raise NotImplementedError