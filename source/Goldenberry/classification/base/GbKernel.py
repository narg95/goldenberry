import numpy as np
from Goldenberry.base.GbDynamicFunction import GbDynamicFunction

class GbKernel(GbDynamicFunction):

    def execute(self, X, Y):
        results = self._dynamic_function_(X, Y)
        for res in results.flat:
            self._update_statistics(res)
        
        return results    

    def cost(self, solutions):
        return self.execute(solutions)

