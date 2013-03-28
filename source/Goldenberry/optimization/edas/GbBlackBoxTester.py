import abc
import numpy as np
import time
from Goldenberry.optimization.base.GbBaseOptimizer import GbBaseOptimizer
from Goldenberry.optimization.base.GbSolution import GbSolution

class GbBlackBoxTester(object):
    __metaclass__ = abc.ABCMeta
    """Optmizers Tester"""
    
    def test(self, optimizer, num_evals):
        if not optimizer.ready():
            return

        run_results = []
        test_results = []
        means=[]
        evals=[]
        stds=[]
        costs=[]
        times=[] 
        
        for run_id in range(num_evals):
            tic= time.time()
            result = optimizer.search()
            toc= time.time()- tic
            eval, argmin, argmax, min, max, mean, std = optimizer.cost_func.statistics()
            run_results.append((result.params, result.cost, eval, toc, mean, std, min, max, argmin, argmax, run_id))
            times.append(toc) 
            means.append(mean)
            stds.append(std)
            costs.append(result.cost)
            evals.append(eval)
            optimizer.reset()

        test_results =(np.max(costs), np.average(costs), np.average(evals), np.average(times), \
                       np.std(costs), np.std(evals), np.std(times), \
                       np.min(costs), np.min(evals), np.min(times), \
                       np.max(evals), np.max(times))

        return run_results, test_results
