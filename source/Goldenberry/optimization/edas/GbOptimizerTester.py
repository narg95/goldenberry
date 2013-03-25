import abc
import cmath, numpy
from Goldenberry.optimization.base.GbBaseOptimizer import GbBaseOptimizer
from Goldenberry.optimization.base.GbSolution import GbSolution

class GbOptimizersTester(object):
    __metaclass__ = abc.ABCMeta
    """Optmizers Tester"""
    listParam = []
    listCals = []
    listMean=[]
    listStdev=[]
    listCost=[]
    num_evals = None
    optimizer =None
    result = None
    evals= argmin= argmax= min= max= mean= stdev=None
    hasFinished=False

    def setup(self, optimizer = None, num_evals=None):
        self.optimizer=optimizer
        self.num_evals=num_evals
        
    def evaluate(self):
        if self.ready():
             i = 0
             while i < self.num_evals:
                 self.result = self.optimizer.search()
                 self.evals, self.argmin, self.argmax, self.min, self.max, self.mean, self.stdev = self.optimizer.cost_func.statistics()
                 self.listParam.append('-'.join([str(self.optimizer.name), str(self.result.params),str(self.result.cost),str(self.evals), str(self.argmin), str(self.argmax), str(self.min), str(self.max), str(self.mean), str(self.stdev)]))
                 self.listMean.append(self.mean)
                 self.listStdev.append(self.stdev)
                 self.listCost.append(self.result.cost)
                 self.optimizer.reset()
                 self.reset()
                 i+=1

             
             self.listCals.append(self.calMean(self.listMean))
             self.listCals.append(self.calStdev(self.listMean))
             self.listCals.append(self.calMaxMin(self.listMean))

             self.listCals.append(self.calMean(self.listStdev))
             self.listCals.append(self.calStdev(self.listStdev))
             self.listCals.append(self.calMaxMin(self.listStdev))

             self.listCals.append(self.calMean(self.listCost))
             self.listCals.append(self.calStdev(self.listCost))
             self.listCals.append(self.calMaxMin(self.listCost))
             self.listCals.append(self.optimizer.name)

             self.hasFinished=True

             #print self.listRun
             #print self.listExperiment
   
    def reset(self):
        self.result = None
        self.evals= self.argmin= self.argmax= self.min= self.max= self.mean= self.stdev=None
        
    def resetList(self):
        self.listCals=[]
        self.listCost=[]
        self.listMean=[]
        self.listParam=[]
        self.listStdev=[]

    def run(self, optimizer, num_evals):
        self.setup(optimizer,num_evals)
        self.evaluate()
             
    def calStdev(self,datalist):
        lista = datalist
        lista2 = []
        A = len(lista)
        suma=0
        varis=0
        if A>1:
            for i in lista:
                suma += i
            p = ((suma+0.0)/(A+0.0))
            for j in range((A)):
                 sumat = (lista[j]-p)**2
                 lista2.append(sumat)
            for k in lista2:
                varis += k
            vari = varis
            va = cmath.sqrt((vari+0.0)/(A+0.0))
            return va
        else:
            print 'error'

    def calMean(self, datalist):
        if len(datalist) > 1:
            mean = numpy.mean(datalist)
            return mean
        else:
            print 'error'

    def calMaxMin(self, datalist):
        lista = datalist
        tam = len(datalist)
        if(tam>1):
            for i in range(1,tam):
                for j in range(0,tam-i):
                    if(lista[j] > lista[j+1]):
                        k = lista[j+1]
                        lista[j+1] = lista[j]
                        lista[j] = k;
            return lista[0],lista[tam-1]
        else:
            print 'error'

    def ready(self):
        return None != self.optimizer

