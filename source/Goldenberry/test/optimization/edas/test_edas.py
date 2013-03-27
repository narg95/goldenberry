from Goldenberry.optimization.edas.Univariate import Cga, Pbil, Tilda
from Goldenberry.optimization.edas.Bivariate import Bmda
from Goldenberry.optimization.base.GbSolution import GbSolution
from Goldenberry.optimization.cost_functions import *
from Goldenberry.optimization.edas.GbBlackBoxTester import GbBlackBoxTester
from Goldenberry.optimization.base.GbCostFunction import GbCostFunction
from unittest import *
import numpy as np

class CgaTest(TestCase):
    """Test class for the Cga algorithm"""
    def test_basic(self):
        cga = Cga()
        cga.setup(10, 20)
        cga.cost_func = GbCostFunction(OneMax)
        result = cga.search()
        self.assertTrue(result.params.all())
        self.assertGreaterEqual(result.cost, 8)

    """Test if the reset function allows a new algorithm execution."""    
    def test_reset(self):
        cga = Cga()
        cga.setup(10, 20)
        cga.cost_func = GbCostFunction(OneMax)
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
        cga.cost_func = GbCostFunction(OneMax)
        self.assertTrue(cga.ready())

class PbilTest(TestCase):
    """Test class for the pbil algorithm"""
    def test_basic(self):
        pbil = Pbil()
        pbil.setup(10, 20)
        pbil.cost_func = GbCostFunction(OneMax)
        result = pbil.search()
        self.assertTrue(result.params.all())
        self.assertGreaterEqual(result.cost, 8)

    """Test if the reset function allows a new algorithm execution."""    
    def test_reset(self):
        pbil = Pbil()
        pbil.setup(10, 20)
        pbil.cost_func = GbCostFunction(OneMax)
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
        pbil.cost_func = GbCostFunction(OneMax)
        self.assertTrue(pbil.ready())

class BmdaTest(TestCase):
    
    """Test the generation of the chi square matrix"""
    def test_shape_calculate_chisquare_matrix(self):
        pop = np.array(([[0, 0, 0, 1], [1, 0, 1, 0], [0, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1],  [1,1,0,0]]))
        chi_matrix = Bmda._get_chisqr_matrix(pop)
        self.assertEqual(chi_matrix.shape, (4,4))
        expected = np.array([[0, 0, 0, 0],[0, 0, 0, 0], [0, 0, 0 , 0],[0, 0, 0, 0]])
        self.assertAlmostEqual(0.0, np.sum((chi_matrix - expected)), places=4)

    """Test the max chi square algorithm"""
    def test_max_chisquare_base(self):
        chi_matrix = np.array([[0.0,4.0,0.0],[4,0.0,5.0],[0.0, 5.0, 0.0]])
        x,y,chi = Bmda._max_chisqr([0,1],[2], chi_matrix)
        self.assertEqual(x, 1)
        self.assertEqual(y, 2)
        self.assertEqual(chi, 5.0)

    """Test the max chi square algorithm"""
    def test_max_chisquare_base_1(self):
        chi_matrix = np.array([[0.0,4.0,0.0],[4,0.0,5.0],[0.0, 5.0, 0.0]])
        x,y,chi = Bmda._max_chisqr([1],[0, 2], chi_matrix)
        self.assertEqual(x, 1)
        self.assertEqual(y, 2)
        self.assertEqual(chi, 5.0)

    """Test the max chi square algorithm"""
    def test_max_chisquare_base_2(self):
        chi_matrix = np.array([[0.0,4.0,0.0],[4,0.0,5.0],[0.0, 5.0, 0.0]])
        x,y,chi = Bmda._max_chisqr([0],[1, 2], chi_matrix)
        self.assertEqual(x, 0)
        self.assertEqual(y, 1)
        self.assertEqual(chi, 4.0)

    def test_generate_graph_all_independent(self):
        """Test that the algorithm generates no graph, all variables independent"""
        pop = np.array(([[0, 0, 0, 1], [1, 0, 1, 0], [0, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1]]))
        roots, _, _ = Bmda.build_graph(pop)
        roots.sort()
        self.assertTrue(np.equal(roots, [0,1,2,3]).all())

    def test_generate_graph_with_dependencies(self):
        """Test that the algorithm generates a graph with two root nodes"""
        pop = np.array(([[0, 0, 1, 0], [1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [1, 1, 0, 0]]))
        roots, _, _ = Bmda.build_graph(pop)
        self.assertEqual(len(roots), 2)
        self.assertTrue(np.equal(roots, 3).any())

    def test_basic_search_onemax(self):
        """Test class for the Bmda algorithm"""
        bmda = Bmda()
        bmda.setup(10, 40)
        bmda.cost_func = GbCostFunction(OneMax)
        result = bmda.search()
        self.assertGreaterEqual(result.params.sum(), 8.0)
        self.assertGreaterEqual(result.cost, 8.0)

    def test_basic_search_zero(self):
        """Test class for the Bmda algorithm"""
        bmda = Bmda()
        bmda.setup(10, 40)
        bmda.cost_func = GbCostFunction(ZeroMax)
        result = bmda.search()
        self.assertTrue((result.params + 1).all())
        self.assertGreaterEqual(result.cost, 8.0)

    def test_basic_search_onemax(self):
        """Test class for the Bmda algorithm"""
        var_size = 10
        bmda = Bmda()
        bmda.setup(var_size, 40)
        bmda.cost_func = GbCostFunction(OneMax)
        result = bmda.search()
        self.assertTrue((result.params + 1).all())
        self.assertGreaterEqual(result.cost, var_size*0.8)
     
    def test_reset(self):
        """Test if the reset function allows a new algorithm execution."""    
        bmda = Bmda()
        bmda.setup(10, 40)
        bmda.cost_func = GbCostFunction(OneMax)
        bmda.search()
        bmda.reset()
        self.assertEqual(bmda.iters, 0)
        self.assertIsNotNone(bmda.distr)
        self.assertEqual(bmda.cost_func.evals, 0)

class TildaTest(TestCase):
    """Test class for the tilda algorithm"""
    def test_basic(self):
        tilda = Tilda()
        tilda.setup(10, 30, learning_rate = 1.0)
        tilda.cost_func = GbCostFunction(OneMax)
        result = tilda.search()
        #self.assertTrue((result.params < 0.1).all())
        #self.assertGreaterEqual(result.cost, 8)

    """Test if the reset function allows a new algorithm execution."""    
    def test_reset(self):
        tilda = Tilda()
        tilda.setup(10, 20)
        tilda.cost_func = GbCostFunction(OneMax)
        result = tilda.search()
        tilda.reset()
        self.assertEqual(tilda.iters, 0)
        self.assertNotEqual(tilda.distr, None)        
        self.assertEqual(tilda.cost_func.evals, 0)
   
    def test_calculate_means_vars(self):
        means = np.array([2.0])
        vars = np.array([9.0])
        acc_means = np.array([1.5])
        acc_vars = np.array([4.0])
        cand_size = 5
        learning_rate = 0.6
        best = GbSolution(np.array([3.0]), float("-Inf"))
        new_means, new_vars = \
            Tilda._estimate_gaussian(means, vars, acc_means, acc_vars, best, cand_size, learning_rate)
        self.assertAlmostEqual(new_means[0], 1.79, 2)
        self.assertAlmostEqual(new_vars[0], 4.026, 2)

    def test_calculate_means_vars_no_best(self):
        means = np.array([2.0])
        vars = np.array([9.0])
        acc_means = np.array([1.5])
        acc_vars = np.array([4.0])
        cand_size = 5
        learning_rate = 0.6
        best = GbSolution(None, float("-Inf"))
        new_means, new_vars = \
            Tilda._estimate_gaussian(means, vars, acc_means, acc_vars, best, cand_size, learning_rate)
        self.assertAlmostEqual(new_means[0], 0.98, 2)
        self.assertAlmostEqual(new_vars[0], 4.026, 2)
    
class OptmizerTesterTest(TestCase):

    def test_basic(self):
        total_runs = 10
        opttester = GbBlackBoxTester()
        optimizer = Cga()
        optimizer.setup(10, 20)
        optimizer.cost_func = GbCostFunction(OneMax)
        run_results, test_results = opttester.test(optimizer, total_runs)
        self.assertEqual(len(run_results), total_runs)
        self.assertEqual(17, len(test_results))

if __name__ == '__main__':
    unittest.main()
