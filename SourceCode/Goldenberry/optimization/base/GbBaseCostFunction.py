import abc
class GbBaseCostFunction:
    __metaclass__ = abc.ABCMeta

    """This is the base class for all the cost functions"""
    def __call__(self, solution):
        return self.cost(solution)

    @abc.abstractmethod
    def cost(self, solution):
        """Gets the cost of a given solution"""
        pass


