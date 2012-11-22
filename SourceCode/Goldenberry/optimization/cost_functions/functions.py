from Goldenberry.optimization.base.GbBaseCostFunction import GbBaseCostFunction

class onemax(GbBaseCostFunction):
    """ One max fitness function"""
    def cost(self, solution):
        return solution.sum(axis=1)

class zero(GbBaseCostFunction):
    """ Zero fitness function """
    def cost(self, solution):
        return -solution.sum(axis=1)