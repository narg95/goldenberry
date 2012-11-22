class GbBaseSolution(object):
    """This class represents a solution with a cost associated."""

    def __init__(self, parameters, cost = 0.0):
        """initializes a new solution."""
        self._parameters = parameters
        self._cost = cost

    def __getitem__(self, i):
        return self._parameters[i]

    @property
    def parameters(self):
        """Gets the set of parameters for the solution."""
        return self._parameters

    @property
    def cost(self):
        """Gets the solution's cost."""
        return self._cost

    @cost.setter
    def cost(self, cost):
        """Sets the cost to the current solution."""
        self._cost = cost

