class GbSolution(object):
    """This class represents a solution with a cost associated."""

    def __init__(self, params, cost = 0.0):
        """initializes a new solution."""
        self._params = params
        self._cost = cost

    def __getitem__(self, i):
        return self._params[i]

    @property
    def params(self):
        """Gets the set of parameters for the solution."""
        return self._params

    @property
    def cost(self):
        """Gets the solution's cost."""
        return self._cost

    def __str__(self):
        return "[cost: " + str(self.cost) + "]\n[parameters:" + str(self.params) + "]"
   