from baseFitness import baseFitness

class onemax(baseFitness):
    """ One max fitness function"""
    def fitness(self, data):
        return data.sum(axis=1)

class zero(baseFitness):
    """ Zero fitness function """
    def fitness(self, data):
        return -data.sum(axis=1)