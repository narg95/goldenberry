from Goldenberry.optimization.base.GbBaseCostFunction import GbBaseCostFunction

class onemax(GbBaseCostFunction):
    """ One max fitness function"""
    def cost(self, solution):
        solution.cost = solution.parameters.sum(axis=1)

class zero(GbBaseCostFunction):
    """ Zero fitness function """
    def cost(self, solution):
        solution.cost =  -solution.parameters.sum(axis=1)