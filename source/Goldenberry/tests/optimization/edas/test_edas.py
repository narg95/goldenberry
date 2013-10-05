from Goldenberry.optimization.edas.Univariate import Cga, Pbil, Tilda, Pbilc
from Goldenberry.optimization.edas.Bivariate import Bmda, DependencyMethod
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
        cga.setup(20)
        cga.cost_func = GbCostFunction(OneMax, var_size = 10)
        result = cga.search()
        self.assertTrue(result.params.all())
        self.assertGreaterEqual(result.cost, 8)

    """Test if the reset function allows a new algorithm execution."""    
    def test_reset(self):
        cga = Cga()
        cga.setup(20)
        cga.cost_func = GbCostFunction(OneMax, var_size = 10)
        cga.search()
        cga.reset()
        self.assertEqual(cga.iters, 0)
        self.assertNotEqual(cga.distr, None)        
        self.assertEqual(cga.cost_func.evals, 0)

    """"Test whether the ready function informs when the 
    algorithm is ready to search."""
    def test_ready(self):
        cga = Cga()
        cga.setup(20)
        self.assertFalse(cga.ready())
        cga.cost_func = GbCostFunction(OneMax, var_size = 10)
        self.assertTrue(cga.ready())

class PbilTest(TestCase):
    """Test class for the pbil algorithm"""
    def test_basic(self):
        pbil = Pbil()
        pbil.setup(20)
        pbil.cost_func = GbCostFunction(OneMax, var_size = 10)
        result = pbil.search()
        self.assertTrue(result.params.all())
        self.assertGreaterEqual(result.cost, 8)

    """Test if the reset function allows a new algorithm execution."""    
    def test_reset(self):
        pbil = Pbil()
        pbil.setup(20)
        pbil.cost_func = GbCostFunction(OneMax, var_size = 10)
        pbil.search()
        pbil.reset()
        self.assertEqual(pbil.iters, 0)
        self.assertNotEqual(pbil.distr, None)        
        self.assertEqual(pbil.cost_func.evals, 0)

    """"Test whether the ready function informs when the 
    algorithm is ready to search."""
    def test_ready(self):
        pbil = Pbil()
        pbil.setup(20)
        self.assertFalse(pbil.ready())
        pbil.cost_func = GbCostFunction(OneMax, var_size = 10)
        self.assertTrue(pbil.ready())

class BmdaTest(TestCase):
    
    """Test the generation of the chi square matrix"""
    def test_shape_calculate_chisquare_matrix(self):
        pop = np.array(([[0, 0, 0, 1], [1, 0, 1, 0], [0, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1],  [1,1,0,0]]))
        chi_matrix = Bmda.build_chi2_dependency_matrix(pop)
        self.assertEqual(chi_matrix.shape, (4,4))
        expected = np.array([[0, 0, 0, 0],[0, 0, 0, 0], [0, 0, 0 , 0],[0, 0, 0, 0]])
        #Test for a total independent chi matrix.
        self.assertTrue((chi_matrix < 3.84).all())

    """Test the max chi square algorithm"""
    def test_max_chisquare_base(self):
        chi_matrix = np.array([[0.0,4.0,0.0],[4,0.0,5.0],[0.0, 5.0, 0.0]])
        x,y,chi = Bmda.get_max_dependency([0,1],[2], chi_matrix)
        self.assertEqual(x, 1)
        self.assertEqual(y, 2)
        self.assertEqual(chi, 5.0)

    """Test the max chi square algorithm"""
    def test_max_chisquare_base_1(self):
        chi_matrix = np.array([[0.0,4.0,0.0],[4,0.0,5.0],[0.0, 5.0, 0.0]])
        x,y,chi = Bmda.get_max_dependency([1],[0, 2], chi_matrix)
        self.assertEqual(x, 1)
        self.assertEqual(y, 2)
        self.assertEqual(chi, 5.0)

    """Test the max chi square algorithm"""
    def test_max_chisquare_base_2(self):
        chi_matrix = np.array([[0.0,4.0,0.0],[4,0.0,5.0],[0.0, 5.0, 0.0]])
        x,y,chi = Bmda.get_max_dependency([0],[1, 2], chi_matrix)
        self.assertEqual(x, 0)
        self.assertEqual(y, 1)
        self.assertEqual(chi, 4.0)

    def test_generate_graph_all_independent(self):
        """Test that the algorithm generates no graph, all variables independent"""
        pop = np.array(([[0, 0, 0, 1], [1, 0, 1, 0], [0, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1]]))
        roots, _, _ = Bmda.build_graph(pop, np.zeros(4), DependencyMethod.chi2_test)
        roots.sort()
        self.assertTrue(np.equal(roots, [0,1,2,3]).all())

    def test_generate_graph_with_dependencies(self):
        """Test that the algorithm generates a graph with two root nodes"""
        pop = np.array(([[0, 0, 1, 0], [1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [1, 1, 0, 0]]))
        roots, _, _ = Bmda.build_graph(pop, np.zeros(4), DependencyMethod.chi2_test)
        self.assertEqual(len(roots), 2)
        self.assertTrue(np.equal(roots, 3).any())

    def test_basic_search_onemax_sim_method(self):
        """Test class for the Bmda algorithm"""
        bmda = Bmda()
        bmda.setup(40, dependency_method = DependencyMethod.sim)
        bmda.cost_func = GbCostFunction(OneMax, var_size = 10)
        result = bmda.search()
        self.assertGreaterEqual(result.params.sum(), 8.0)
        self.assertGreaterEqual(result.cost, 8.0)

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
        bmda.setup(40)
        bmda.cost_func = GbCostFunction(ZeroMax, var_size = 10)
        result = bmda.search()
        self.assertTrue((result.params + 1).all())
        self.assertGreaterEqual(result.cost, 8.0)

    def test_basic_search_onemax(self):
        """Test class for the Bmda algorithm"""
        var_size = 10
        bmda = Bmda()
        bmda.setup(40)
        bmda.cost_func = GbCostFunction(OneMax, var_size = var_size)
        result = bmda.search()
        self.assertTrue((result.params + 1).all())
        self.assertGreaterEqual(result.cost, var_size*0.8)
     
    def test_reset(self):
        """Test if the reset function allows a new algorithm execution."""    
        bmda = Bmda()
        bmda.setup(40)
        bmda.cost_func = GbCostFunction(OneMax, var_size = 10)
        bmda.search()
        bmda.reset()
        self.assertEqual(bmda.iters, 0)
        self.assertIsNotNone(bmda.distr)
        self.assertEqual(bmda.cost_func.evals, 0)

class TildaTest(TestCase):
    """Test class for the tilda algorithm"""
    def test_basic(self):
        tilda = Tilda()
        tilda.setup(40, learning_rate = 0.3, max_evals = 10000)
        tilda.cost_func = GbCostFunction(OneMax, var_size = 10)
        result = tilda.search()
        self.assertGreaterEqual(result.cost, 8.0)

    """Test if the reset function allows a new algorithm execution."""    
    def test_reset(self):
        tilda = Tilda()
        tilda.setup(20)
        tilda.cost_func = GbCostFunction(OneMax, var_size = 10)
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
        self.assertAlmostEqual(new_means[0], 1.16, 2)
        self.assertAlmostEqual(new_vars[0], 4.2681, 2)

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
        self.assertAlmostEqual(new_means[0], 1.16, 2)
        self.assertAlmostEqual(new_vars[0], 4.2681, 2)

class PbilcTest(TestCase):
    """Test class for the pbilc algorithm"""
    def test_basic(self):
        pbilc = Pbilc()
        pbilc.setup(30, learning_rate = 0.5, max_evals = 1000)
        pbilc.cost_func = GbCostFunction(OneMax, var_size = 10)
        result = pbilc.search()        
        self.assertGreaterEqual(result.cost, 8)

    """Test if the reset function allows a new algorithm execution."""    
    def test_reset(self):
        pbilc = Pbilc()
        pbilc.setup(20)
        pbilc.cost_func = GbCostFunction(OneMax, var_size = 10)
        result = pbilc.search()
        pbilc.reset()
        self.assertEqual(pbilc.iters, 0)
        self.assertNotEqual(pbilc.distr, None)        
        self.assertEqual(pbilc.cost_func.evals, 0)    

class OptmizerTesterTest(TestCase):

    def test_basic(self):
        total_runs = 10
        opttester = GbBlackBoxTester()
        optimizer = Cga()
        optimizer.setup(20)
        optimizer.cost_func = GbCostFunction(OneMax, var_size = 10)
        run_results, test_results = opttester.test(optimizer, total_runs)
        self.assertEqual(len(run_results), total_runs)
        self.assertEqual(12, len(test_results))

if __name__ == '__main__':
    unittest.main()
