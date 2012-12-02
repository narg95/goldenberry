import numpy as np
from Goldenberry.optimization.edas.distributions import *
from Goldenberry.optimization.edas.BaseEda import BaseEda
from Goldenberry.optimization.base.GbSolution import *

class Bmda(BaseEda):
    """Bivariate marginal distribution algorithm."""

    _pop_size = None
    _vars_size = None
    _cost_function = None
    _distribution = None
    _max_iters = None
    _iters = None

    def setup(self, cost_function, varsize, popsize, maxiters = None):
        """Configure a Cga instance"""
        self._pop_size = popsize
        self._vars_size = varsize
        self._cost_function = cost_function
        self._distribution = Binomial(params = np.tile(0.5,(1,varsize)))
        self._max_iters = maxiters
        self._iters = 0

    def result_distribution(self):
        """Provides the final estimated distribution."""
        pass

    def ready(self):
        """Informs if the algorithm is ready to execute."""
        pass


