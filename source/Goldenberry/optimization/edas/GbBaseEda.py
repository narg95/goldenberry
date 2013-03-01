import abc
from Goldenberry.optimization.base.GbBaseOptimizer import GbBaseOptimizer

class GbBaseEda(GbBaseOptimizer):
    """This class represents a solution with a cost associated."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def distr(self):
        """Provides the final estimated distribution."""
        pass

    @abc.abstractmethod
    def update_distribution(self):
        """Updates the current distribution."""
        pass

    @abc.abstractmethod
    def update_candidates(self):
        """Generates the new generation of candidate solutions."""
        pass

    @abc.abstractmethod
    def hasFinished(self):
        pass
