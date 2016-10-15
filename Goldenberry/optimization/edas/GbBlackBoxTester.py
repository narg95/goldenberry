import abc
import numpy as np
import time
from Goldenberry.optimization.base.GbBaseOptimizer import GbBaseOptimizer
from Goldenberry.optimization.base.GbSolution import GbSolution

class GbBlackBoxTester(object):
    __metaclass__ = abc.ABCMeta
    """Optmizers Tester"""
    
    def test(self, optimizer, num_evals, callback = None):
        if not optimizer.ready():
            return

        run_results = []
        test_results = []
        means=[]
        evals=[]
        stds=[]
        costs=[]
        times=[] 
        candidates = []
        
        for run_id in range(num_evals):
            if callback is not None:
                optimizer.callback_func = lambda best, progress : callback(best, (progress + run_id)/float(num_evals))

            tic= time.time()
            result = optimizer.search()
            toc= time.time()- tic
            eval, argmin, argmax, min, max, mean, std = optimizer.cost_func.statistics()
            run_results.append((result.params, result.cost, eval, toc, mean, std, min, max, argmin, argmax, run_id, result.roots, result.children, get_tree(result.children)))
            candidates.append(result)
            times.append(toc) 
            means.append(mean)
            stds.append(std)
            costs.append(result.cost)
            evals.append(eval)

        test_results =(np.max(costs), np.average(costs), np.average(evals), np.average(times), \
                       np.std(costs), np.std(evals), np.std(times), \
                       np.min(costs), np.min(evals), np.min(times), \
                       np.max(evals), np.max(times))

        return run_results, test_results, candidates
    
def get_tree(children):
    tree = [None]*len(children)
    for parent, child_list in enumerate(children):
        for child in child_list:
            tree[child] = parent
    return tree