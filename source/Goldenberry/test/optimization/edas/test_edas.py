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
    
    """Test the generation of the chi square matrix"""
    def test_shape_calculate_chisquare_matrix(self):
        pop = np.array(([[0, 0, 0, 1], [1, 0, 1, 0], [0, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1]]))
        chi_matrix = Bmda.calculate_chisquare_matrix(pop)
        self.assertEqual(chi_matrix.shape, (4,4))
        self.assertEquals(np.not_equal(chi_matrix,0.0).sum(), 4)

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
        bmda.setup(onemax(), 10, 30)
        result = bmda.search()
        self.assertTrue(result.params.all())
        self.assertEqual(result.cost, 10)

    """Test class for the Bmda algorithm"""
    def test_basic_search_zero(self):
        bmda = Bmda()
        bmda.setup(zero(), 10, 30)
        result = bmda.search()
        self.assertTrue(~result.params.all())
        self.assertEqual(result.cost, 0)

if __name__ == '__main__':
    unittest.main()
