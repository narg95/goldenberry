import abc
from Goldenberry.optimization.base.GbBaseOptimizer import GbBaseOptimizer

class BaseEda(GbBaseOptimizer):
    """This class represents a solution with a cost associated."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def distr(self):
        """Provides the final estimated distribution."""
        pass

    @abc.abstractmethod
    def ready(self):
        """Informs if the algorithm is ready to execute."""
        pass

    @abc.abstractmethod
    def update_distribution(self):
        """Updates the current distribution."""
        pass

