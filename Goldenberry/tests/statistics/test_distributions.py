from Goldenberry.statistics.distributions import *
import numpy as np
from scipy import stats

from unittest import *

class BinomialTest(TestCase):
    
    def test_basic(self):
        dist = Binomial(10)        

    def test_vars_size(self):
        n = 10
        dist = Binomial(n = n)
        n1, p1 = dist.n, dist.p
        self.assertEqual(n1, n)
        self.assertTrue(p1.all())

    def test_parameters(self):
        n = 10
        p = np.tile(0.5,(1,n))
        dist = Binomial(p = p)
        n1, p1 = dist.n, dist.p
        self.assertTrue(np.all(e == 0.5 for e in p1))
        self.assertEqual(n1, n)

    def test_sample(self):
        vars_size = 10
        dist = Binomial(p = np.array([0.0, 0.5, 1]))
        sample = dist.sample(1)
        self.assertTrue(\
            sample[0, 0] == 0.0 and \
            sample[0, 1] <= 1.0 and \
            sample[0, 1] >= 0.0 and \
            sample[0, 2] == 1.0)

class BivariateBinomialTest(TestCase):
    
    def test_basic(self):
        n = 10
        dist = BivariateBinomial(n)
        self.assertIsNotNone(dist)
        n1, p, cond_props, children = dist.n, dist.p, dist.cond_props, dist.children 
        self.assertEqual(n1, n)
        self.assertTrue(np.all(e == 0.5 for e in p))
        self.assertTrue(np.all(e == [] for e in cond_props))
        self.assertTrue(np.all(e == [] for e in children))

    def test_independency_sample(self):
        n = 10
        dist = BivariateBinomial(n)
        samples = dist.sample(20)
        self.assertTrue(samples.shape == (20,10))
        self.assertTrue(np.equal(dist.roots, range(n)).all())

    def test_sample_chain_all_ones(self):
        n = 5
        p = np.ones((1,n))
        cond_props = [[] if i == 0 else np.array([1, 1]) for i in xrange(n)]
        children = [[] if i == n-1 else [i + 1] for i in xrange(n)]
        dist = BivariateBinomial(p = p, cond_props = cond_props, children = children)
        samples = dist.sample(20)
        self.assertTrue(np.all(samples == 1.0))
    
        #todo Refactor with new strategy    
    def test_sample_chain_ones_and_zeros_interleaving(self):
        n = 10
        p = np.array([i%2 for i in range(n)])
        p.shape = (1,10)
        cond_props = [[] if i == 0 else np.array([i%2, i%2]) for i in xrange(n)]
        children = [[] if i == n-1 else [i + 1] for i in xrange(n)]
        dist = BivariateBinomial(p = p, cond_props = cond_props, children = children)
        samples = dist.sample(20)
        evenx = range(0,n,2)
        oddx = range(1,n,2)
        self.assertTrue(np.all(samples[:, evenx] == 0))
        self.assertTrue(np.all(samples[:, oddx] == 1.0))
    
class BinomialContingencyTableTest(TestCase):
    
    def test_basic(self):
        X = np.array([[0],[1]])
        Y = np.array([[0, 1, 0], [0, 0, 1]])
        ctable = BinomialContingencyTable(X, Y)
        self.assertEqual(ctable.n, 2)
        self.assertIsNotNone(ctable.table)
        self.assertEqual(ctable.pys.shape, (3,))
        self.assertEqual(ctable.pxys.shape, (2,6))

    def test_contingency_table_basic(self):
        X = np.array([[0],[1],[0],[1]])
        Y = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0]])
        ctable = BinomialContingencyTable(X, Y)
        expected = np.array([[1, 1, 2, 0, 1, 1], [1, 1, 0, 2, 1, 1]])
        self.assertTrue(np.all((ctable.table - expected) == 0))

    def test_chisquare_test(self):
        X = np.array([[0],[1],[0],[1],[0],[1]])
        Y = np.array([[1,0,1], [0,0,1], [0,1,0], [0,1,0],[1,1,1], [0,0,1]])
        ctable = BinomialContingencyTable(X, Y)
        chi = ctable.chisquare()
        expected = np.array([3, 0.6667, 5.54e-32])
        self.assertAlmostEqual(np.sum(chi - expected ), 0.0, places=4)

    def test_chisquare_mutual_inf_test(self):
        X = np.array([[0],[1],[0],[1],[0],[1]])
        Y = np.array([[1,0,1], [0,0,1], [0,1,0], [0,1,0],[1,1,1], [0,0,1]])
        ctable = BinomialContingencyTable(X, Y)
        chi, mi = ctable.chi_and_mutual_inf()
        expected_chi = np.array([3, 0.6667, 0])
        expected_mi = np.array([4.591479e-01, 8.170417e-02, 0])
        self.assertAlmostEqual(np.sum(chi - expected_chi ), 0.0, places=4)
        self.assertAlmostEqual(np.sum(mi - expected_mi), 0.0, places=4)

    def test_sim_test(self):
        X = np.array([[0],[1],[0],[1],[0],[1]])
        Y = np.array([[1,0,1], [0,0,1], [0,1,0], [0,1,0],[1,1,1], [0,0,1]])
        ctable = BinomialContingencyTable(X, Y)
        sim = ctable.sim()
        expected = np.array([4.20917e-01,   4.78609e-02,  0])
        self.assertAlmostEqual(np.sum(sim - expected), 0.0, places=4)
        
    def test_join_and_conditional_probability(self):
        X = np.array([[1],[0],[1],[1],[1],[1]])
        Y = np.array([[1],[0],[0],[0],[0],[1]])
        ctable = BinomialContingencyTable(X, Y)
        self.assertEquals(ctable.px, 5.0/6.0)
        self.assertEquals(ctable.pys, [2.0/6.0])
        self.assertTrue(np.equal(ctable.pxys, [[1/6.0, 0.0],[0.5, 2/6.0]]).all())
        self.assertTrue(((ctable.PyGx - np.array([[1.0, 0.0],[0.6, 0.4]])) < 0.001).all())       

    def test_conditional_NAN_probability(self):
        X = np.array([[1],[1],[1],[1],[1],[1]])
        Y = np.array([[0],[0],[0],[1],[1],[1]])
        ctable = BinomialContingencyTable(X, Y)
        self.assertTrue(((ctable.PyGx - np.array([[0.0, 0.0],[0.5, 0.5]])) < 0.001).all())  

class GaussianTest(TestCase):
    
    def test_basic(self):
        n = 10
        dist = Gaussian(n = n)
        means, stdevs = dist.means, dist.stdevs
        self.assertTrue(np.equal(means, np.zeros(n)).all())
        self.assertTrue(np.equal(stdevs, np.ones(n)).all())
        self.assertEqual(dist.n, n)

    def stdevs(self):
        n = 10
        means, stdevs = np.zeros(n), np.ones(n)
        dist = Gaussian(means = means, stdevs = stdevs)
        p_means, p_stdevs = dist.means, dist.stdevs
        self.assertTrue(np.equal(p_means, means))
        self.assertTrue(np.equal(p_stdevs, stdevs))
        self.assertEqual(dist.n, n)

    def test_sample(self):
        n = 10
        means, stdevs = np.tile(0.5, n), np.ones(n)
        dist = Gaussian(means = means, stdevs = stdevs)
        samples = dist.sampletr(20, 0.0, 1.0)
        self.assertTrue((samples < 1.0).all())
        self.assertTrue((samples > 1.0).all())

    def test_sample(self):
        dist = Gaussian(means = np.array([0.0, 1.0, 2.0]), stdevs = np.array([0.0, 0.0, 0.1]))
        sample = dist.sample(1)
        self.assertTrue(\
            sample[0, 0] == 0.0 and \
            sample[0, 1] <= 1.0 and \
            sample[0, 1] >= 1.0 )    

class GaussianTruncTest(TestCase):
    
    def test_basic(self):
        n = 10
        dist = GaussianTrunc(n = n, low = 1.0, high = 2.0)
        self.assertEqual(dist.low, 1.0)   
        self.assertEqual(dist.high, 2.0)

    def test_sample(self):
        n = 10
        means, stdevs = np.tile(0.5, n), np.ones(n)
        dist = GaussianTrunc(means = means, stdevs = stdevs)
        samples = dist.sample(100)
        self.assertTrue((samples <= 1.0).all())
        self.assertTrue((samples >= 0.0).all())

    def test_sample_one_to_two(self):
        n = 10
        means, stdevs = np.tile(0.5, n), np.ones(n)
        dist = GaussianTrunc(means = means, stdevs = stdevs, low = 1.0, high = 2.0)
        samples = dist.sample(100)
        self.assertTrue((samples <= 2.0).all())
        self.assertTrue((samples >= 1.0).all())