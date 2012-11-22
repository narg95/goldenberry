from Goldenberry.optimization.base.GbBaseOptimizer import GbBaseOptimizer

class BaseEda(GbBaseOptimizer):
    """This class represents a solution with a cost associated."""

    def result_distribution(self):
        """Provides the final estimated distribution."""
        pass

    def ready(self):
        """Informs if the algorithm is ready to execute."""
        pass

