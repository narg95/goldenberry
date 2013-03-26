import abc
import numpy as np
from Goldenberry.optimization.base.GbBaseOptimizer import GbBaseOptimizer
from Goldenberry.optimization.base.GbSolution import GbSolution

class GbOptimizersTester(object):
    __metaclass__ = abc.ABCMeta
    """Optmizers Tester"""
    
    def run(self, optimizer, num_evals):
        if not optimizer.ready():
            return

        run_results = []
        test_results = []
        means=[]
        evals=[]
        vars=[]
        costs=[] 
        for run_id in range(num_evals):
            result = optimizer.search()
            eval, argmin, argmax, min, max, mean, var = optimizer.cost_func.statistics()
            run_results.append((run_id, result.params, result.cost, eval, argmin, argmax, min, max, mean, var))
             
            means.append(mean)
            vars.append(var)
            costs.append(result.cost)
            evals.append(eval)
            optimizer.reset()
             
        test_results =(np.mean(evals), np.var(evals), np.min(evals), np.max(evals), \
                       np.mean(costs), np.var(costs), np.min(costs), np.max(costs), \
                       np.mean(means), np.var(means), np.min(means), np.max(means), \
                       np.mean(vars), np.var(vars), np.min(vars), np.max(vars))
        
        return run_results, test_results