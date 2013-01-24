from Goldenberry.statistics.distributions import *
import numpy as np

from unittest import *

class BinomialTest(TestCase):
    
    def test_basic(self):
        dist = Binomial(10)        

    def test_vars_size(self):
        n = 10
        dist = Binomial(n = n)
        n1, p1 = dist.parameters
        self.assertEqual(n1, n)
        self.assertTrue(p1.all())

    def test_parameters(self):
        n = 10
        p = np.tile(0.5,(1,n))
        dist = Binomial(p = p)
        n1, p1 = dist.parameters
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
        n1, p, cond_props, children = dist.parameters 
        self.assertEqual(n1, n)
        self.assertTrue(np.all(e == 0.5 for e in p))
        self.assertTrue(np.all(e == [] for e in cond_props))
        self.assertTrue(np.all(e == [] for e in children))

    def test_independency_sampling(self):
        n = 10
        dist = BivariateBinomial(n)
        samples = dist.sample(20)
        self.assertTrue(samples.shape == (20,10))

    def test_sampling_chain_all_ones(self):
        n = 5
        p = np.ones((1,n))
        cond_props = [[] if i == 0 else np.array([1, 1]) for i in xrange(n)]
        children = [[] if i == n-1 else [i + 1] for i in xrange(n)]
        dist = BivariateBinomial(p = p, cond_props = cond_props, children = children)
        samples = dist.sample(20)
        self.assertTrue(np.all(samples == 1.0))
    
        #todo Refactor with new strategy    
    def test_sampling_chain_ones_and_zeros_interleaving(self):
        n = 10
        p = np.array([i%2 for i in range(n)])
        p.shape = (1,10)
        pyGx = np.array([[(i + 1)%2 for i in range(n-1)], [(i + 1)%2 for i in range(n-1)]])
        edges = [(x, x+1) for x in range(n-1)]
        dist = BivariateBinomial(p = p, cond_props = pyGx, children = edges)
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
        X = np.array([[0],[1],[0],[1]])
        Y = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0]])
        ctable = BinomialContingencyTable(X, Y)
        chi = ctable.chisquare()
        expected = np.array([0, 4, 0])
        self.assertTrue(np.all((chi - expected )== 0))