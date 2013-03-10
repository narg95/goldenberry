from Goldenberry.optimization.edas.Univariated import Cga, Pbil
from Goldenberry.optimization.edas.Bivariated import Bmda
from Goldenberry.optimization.cost_functions.functions import *
from unittest import *
import numpy as np

class CgaTest(TestCase):
    """Test class for the Cga algorithm"""
    def test_basic(self):
        cga = Cga()
        cga.setup(10, 20)
        cga.cost_func = Onemax()
        result = cga.search()
        self.assertTrue(result.params.all())
        self.assertGreaterEqual(result.cost, 8)

    """Test if the reset function allows a new algorithm execution."""    
    def test_reset(self):
        cga = Cga()
        cga.setup(10, 20)
        cga.cost_func = Onemax()
        cga.search()
        cga.reset()
        self.assertEqual(cga.iters, 0)
        self.assertNotEqual(cga.distr, None)        
        self.assertEqual(cga.cost_func.evals, 0)

    """"Test whether the ready function informs when the 
    algorithm is ready to search."""
    def test_ready(self):
        cga = Cga()
        cga.setup(10, 20)
        self.assertFalse(cga.ready())
        cga.cost_func = Onemax()
        self.assertTrue(cga.ready())

class PbilTest(TestCase):
    """Test class for the pbil algorithm"""
    def test_basic(self):
        pbil = Pbil()
        pbil.setup(10, 20)
        pbil.cost_func = Onemax()
        result = pbil.search()
        self.assertTrue(result.params.all())
        self.assertGreaterEqual(result.cost, 8)

    """Test if the reset function allows a new algorithm execution."""    
    def test_reset(self):
        pbil = Pbil()
        pbil.setup(10, 20)
        pbil.cost_func = Onemax()
        pbil.search()
        pbil.reset()
        self.assertEqual(pbil.iters, 0)
        self.assertNotEqual(pbil.distr, None)        
        self.assertEqual(pbil.cost_func.evals, 0)

    """"Test whether the ready function informs when the 
    algorithm is ready to search."""
    def test_ready(self):
        pbil = Pbil()
        pbil.setup(10, 20)
        self.assertFalse(pbil.ready())
        pbil.cost_func = Onemax()
        self.assertTrue(pbil.ready())

class BmdaTest(TestCase):
    
    """Test the generation of the chi square matrix"""
    def test_shape_calculate_chisquare_matrix(self):
        pop = np.array(([[0, 0, 0, 1], [1, 0, 1, 0], [0, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1],  [1,1,0,0]]))
        chi_matrix = Bmda.calculate_chisquare_matrix(pop)
        self.assertEqual(chi_matrix.shape, (4,4))
        expected = np.array([[0, 0, 0, 0],[0, 0, 0, 0], [0, 0, 0 , 0],[0, 0, 0, 0]])
        self.assertAlmostEqual(0.0, np.sum((chi_matrix - expected)), places=4)

    """Test the max chi square algorithm"""
    def test_max_chisquare_base(self):
        chi_matrix = np.array([[0.0,4.0,0.0],[4,0.0,5.0],[0.0, 5.0, 0.0]])
        x,y,chi = Bmda.max_chisquare([0,1],[2], chi_matrix)
        self.assertEqual(x, 1)
        self.assertEqual(y, 2)
        self.assertEqual(chi, 5.0)

    """Test the max chi square algorithm"""
    def test_max_chisquare_base_1(self):
        chi_matrix = np.array([[0.0,4.0,0.0],[4,0.0,5.0],[0.0, 5.0, 0.0]])
        x,y,chi = Bmda.max_chisquare([1],[0, 2], chi_matrix)
        self.assertEqual(x, 1)
        self.assertEqual(y, 2)
        self.assertEqual(chi, 5.0)

    """Test the max chi square algorithm"""
    def test_max_chisquare_base_2(self):
        chi_matrix = np.array([[0.0,4.0,0.0],[4,0.0,5.0],[0.0, 5.0, 0.0]])
        x,y,chi = Bmda.max_chisquare([0],[1, 2], chi_matrix)
        self.assertEqual(x, 0)
        self.assertEqual(y, 1)
        self.assertEqual(chi, 4.0)

    """Test that the algorithm generates no graph, all variables independent"""
    def test_generate_graph_all_independent(self):
        pop = np.array(([[0, 0, 0, 1], [1, 0, 1, 0], [0, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1]]))
        roots, _, _ = Bmda.generate_graph(pop)
        roots.sort()
        self.assertTrue(np.equal(roots, [0,1,2,3]).all())

    """Test that the algorithm generates a graph with two root nodes"""
    def test_generate_graph_with_dependencies(self):
        pop = np.array(([[0, 0, 1, 0], [1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [1, 1, 0, 0]]))
        roots, _, _ = Bmda.generate_graph(pop)
        self.assertEqual(len(roots), 2)
        self.assertTrue(np.equal(roots, 3).any())

    """Test class for the Bmda algorithm"""
    def test_basic_search_onemax(self):
        bmda = Bmda()
        bmda.setup(10, 40)
        bmda.cost_func = Onemax()
        result = bmda.search()
        self.assertGreaterEqual(result.params.sum(), 8.0)
        self.assertGreaterEqual(result.cost, 8.0)

    """Test class for the Bmda algorithm"""
    def test_basic_search_zero(self):
        bmda = Bmda()
        bmda.setup(10, 40)
        bmda.cost_func = Zeromax()
        result = bmda.search()
        self.assertTrue((result.params + 1).all())
        self.assertGreaterEqual(result.cost, 8.0)

    """Test class for the Bmda algorithm"""
    def test_basic_search_cond_onemax(self):
        var_size = 10
        bmda = Bmda()
        bmda.setup(var_size, 40)
        bmda.cost_func = CondOnemax()
        result = bmda.search()
        self.assertTrue((result.params + 1).all())
        self.assertGreaterEqual(result.cost, var_size*0.8)
     
    """Test if the reset function allows a new algorithm execution."""    
    def test_reset(self):
        bmda = Bmda()
        bmda.setup(10, 40)
        bmda.cost_func = Onemax()
        bmda.search()
        bmda.reset()
        self.assertEqual(bmda.iters, 0)
        self.assertIsNotNone(bmda.distr)
        self.assertEqual(bmda.cost_func.evals, 0)

if __name__ == '__main__':
    unittest.main()
