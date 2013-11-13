class GbSolution(object):
    """This class represents a solution with a cost associated."""

    def __init__(self, params, cost = 0.0, roots = None, children = None):
        """initializes a new solution."""
        self.params = params
        self.cost = cost
        self.roots = roots
        self.children = children
        if None is roots and None is not params:
            self.roots = range(len(params))
            self.children = [[] for i in self.roots]

    def __getitem__(self, i):
        return self.params[i]    

    def __str__(self):
        return "[cost: " + str(self.cost) + "]\n[parameters:" + str(self.params) + "]\n [roots:" + str(self.roots) + "]\n [children:" + str(self.children) + "]"
   