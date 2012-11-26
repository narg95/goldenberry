from Goldenberry.optimization.base.GbBaseCostFunction import GbBaseCostFunction

class onemax(GbBaseCostFunction):
    """ One max"""
    def cost(self, solution):
        return solution.sum(axis=1)

    def name(self):
        """Gets name"""
        return "One max"

class zero(GbBaseCostFunction):
    """ Zero cost"""
    def cost(self, solution):
        return -solution.sum(axis=1)

    def name(self):
        """Gets name"""
        return "Zero function"