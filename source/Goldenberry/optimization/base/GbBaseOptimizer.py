class GbBaseOptimizer(object):
    """Base class for optimization algorithms."""

    def setup(self, cost_function, **kwargs):
        """Setup the optimizer algorithm."""
        pass

    def search(self):
        """Search for a optimal solution"""
        pass