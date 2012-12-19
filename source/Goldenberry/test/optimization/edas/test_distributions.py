from Goldenberry.optimization.edas.distributions import *

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
        n1, p, pyGx, edges = dist.parameters 
        self.assertEqual(n1, n)
        self.assertTrue(np.all(e == 0.5 for e in p))
        self.assertTrue(np.all(e == 0.5 for e in pyGx))
        self.assertTrue(edges == [])

    def test_sampling(self):
        n = 10
        dist = BivariateBinomial(n)
        samples = dist.sample(20)
        self.assertTrue(samples.shape == (20,10))

    def test_sampling_only_ones_no_dependencies(self):
        n = 10
        p = np.ones((1,n))
        pyGx = np.matrix([np.zeros(n-1), np.ones(n-1)])
        edges = [(x, x+1) for x in range(n-1)]
        dist = BivariateBinomial(p = p, pyGx = pyGx, edges = edges)
        samples = dist.sample(20)
        self.assertTrue(np.all(samples == 1.0))
        
    def test_sampling_ones_and_zerps_interleaving(self):
        n = 10
        p = np.array([i%2 for i in range(n)])
        p.shape = (1,10)
        pyGx = np.array([[(i + 1)%2 for i in range(n-1)], [(i + 1)%2 for i in range(n-1)]])
        edges = [(x, x+1) for x in range(n-1)]
        dist = BivariateBinomial(p = p, pyGx = pyGx, edges = edges)
        samples = dist.sample(20)
        # TODO:  update the assert 
        self.assertTrue(np.all(samples == 1.0))
        

