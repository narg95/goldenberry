class GbBaseCostFunction:
    """This is the base class for all the cost functions"""
    def __call__(self, solution):
        return self.cost(solution)

    def cost(self, solution):
        """Gets the cost of a given solution"""
        pass


