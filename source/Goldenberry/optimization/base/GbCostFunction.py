import numpy as np
from Goldenberry.base.GbDynamicFunction import GbDynamicFunction

class GbCostFunction(GbDynamicFunction):

    def execute(self, solutions):
        """Gets the cost of a given solution and calculate the statistics."""
        costs = np.zeros(solutions.shape[0])
        for idx, solution in enumerate(solutions):
            costs[idx] = super(GbCostFunction, self).execute(solution)

        return costs

    def cost(self, solutions):
        return self.execute(solutions)
