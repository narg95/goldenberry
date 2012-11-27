class GbBaseOptimizer(object):
    """Base class for optimization algorithms."""

    def setup(self, cost_function, **kwargs):
        """Setup the optimizer algorithm."""
        raise NotImplementedError

    def search(self):
        """Search for a optimal solution"""
        raise NotImplementedError