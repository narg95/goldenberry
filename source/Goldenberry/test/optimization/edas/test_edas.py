from Goldenberry.optimization.edas.Cga import Cga
from Goldenberry.optimization.edas.Bmda import Bmda
from Goldenberry.optimization.cost_functions.functions import *
from unittest import *
import numpy as np

class CgaTest(TestCase):
    """Test class for the Cga algorithm"""
    def test_basic(self):
        cga = Cga()
        cga.setup(onemax(), 10, 20)
        result = cga.search()
        self.assertTrue(result.params.all())
        self.assertEqual(result.cost, 10)

class BmdaTest(TestCase):
    
    def test_shape_calculate_chisquare_matrix(self):
        pop = np.array(([[0, 0, 0, 1], [1, 0, 1, 0], [0, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1]]))
        cond_props= [[] for i in xrange(4)]
        chi_matrix = Bmda.calculate_chisquare_matrix(pop, cond_props)
        self.assertEqual(chi_matrix.shape, (4,4))
        self.assertEquals(np.not_equal(chi_matrix,0.0).sum(), 4)

    """Test class for the Bmda algorithm"""
    def basic_search(self):
        bmda = Bmda()
        bmda.setup(onemax(), 10, 20)
        result = bmda.search()
        self.assertTrue(result.params.all())
        self.assertEqual(result.cost, 10)

if __name__ == '__main__':
    unittest.main()
