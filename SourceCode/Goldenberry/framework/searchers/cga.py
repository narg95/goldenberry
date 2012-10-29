from MLFw.searchers import *
from MLFw.searchers.baseSearcher import *

class cga(baseSearcher):
    """Compact Genetic Algorithm"""
    
    def config(self, fitness, varsize, popsize, maxiters = None):
        self.popsize = popsize
        self.varsize = varsize
        self.fitness = fitness
        self.margprob = tile(0.5,(1,varsize))
        self.maxiters = maxiters
        self.iters = 0

    def find(self):
        while not self.hasFinished():
            self.iters += 1
            pop = sample(self.margprob, 2, self.varsize)
            winner, losser = self.compete(pop)
            self.updateModel(winner, losser)
        return sample(self.margprob, 1, self.varsize)

    def isReady(self):
        return self.popsize is not None and\
               self.varsize is not None and\
               self.fitness is not None

    def hasFinished(self):
        finish = not (self.maxiters is None) and self.iters > self.maxiters
        if finish:
            return True
        return (((1 - self.margprob) < 0.01) | (self.margprob < 0.01)).all()
    
    def compete(self, pop):
        minindx = argmin(self.fitness.fitness(pop))
        maxindx = 1 if minindx == 0 else 0
        return  pop[maxindx], pop[minindx]

    def updateModel(self, winner, losser):
        for x in range(0, self.varsize):
            if 1.0 >= self.margprob[0,x] >= 0.0:
                self.margprob[0,x] += (winner[x] - losser[x])/float(self.popsize)
