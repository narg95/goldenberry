class GbSolution(object):
    """This class represents a solution with a cost associated."""

    def __init__(self, params, cost = 0.0):
        """initializes a new solution."""
        self.params = params
        self.cost = cost

    def __getitem__(self, i):
        return self._params[i]    

    def __str__(self):
        return "[cost: " + str(self.cost) + "]\n[parameters:" + str(self.params) + "]"
   