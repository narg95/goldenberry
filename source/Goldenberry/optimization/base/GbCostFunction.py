import numpy as np
from Goldenberry.base.GbDynamicFunction import GbDynamicFunction

class GbCostFunction(GbDynamicFunction):

    def __init__(self, func = None, script = None, var_size = 10):
        super(GbCostFunction, self).__init__(func, script)
        self.var_size = var_size        

    def execute(self, solutions):
        """Gets the cost of a given solution and calculate the statistics."""
        costs = np.zeros(solutions.shape[0])
        for idx, solution in enumerate(solutions):
            costs[idx] = super(GbCostFunction, self).execute(solution)

        return costs

    def cost(self, solutions):
        return self.execute(solutions)
