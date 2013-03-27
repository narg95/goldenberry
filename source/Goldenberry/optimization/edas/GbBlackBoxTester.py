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
        vars=[]
        costs=[]
        times=[] 
        
        for run_id in range(num_evals):
            start= time.time()
            result = optimizer.search()
            total_time= time.time()-start
            eval, argmin, argmax, min, max, mean, var = optimizer.cost_func.statistics()
            run_results.append((run_id, result.params, result.cost, eval, argmin, argmax, min, max, mean, var,total_time))
            times.append(total_time) 
            means.append(mean)
            vars.append(var)
            costs.append(result.cost)
            evals.append(eval)
            optimizer.reset()
            total_time=0
            start=0
        test_results =(np.mean(evals), np.var(evals), np.min(evals), np.max(evals), \
                       np.mean(costs), np.var(costs), np.min(costs), np.max(costs), \
                       np.mean(means), np.var(means), np.min(means), np.max(means), \
                       np.mean(vars), np.var(vars), np.min(vars), np.max(vars), np.sum(times))
        
        return run_results, test_results
