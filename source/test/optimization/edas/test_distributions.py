from Goldenberry.optimization.edas.distributions import *

from unittest import *

class BinomialTest(TestCase):
    
    def test_basic(self):
        dist = Binomial(10)        

    def test_vars_size(self):
        vars_size = 10
        dist = Binomial(vars_size = vars_size)
        self.assertEqual(dist.vars_size, vars_size)
        self.assertFalse(dist.parameters.any())

    def test_parameters(self):
        vars_size = 10
        params = np.tile(0.5,(1,vars_size))
        dist = Binomial(params = params)
        self.assertTrue(np.all(e == 0.5 for e in dist.parameters))
        self.assertEqual(dist.vars_size, vars_size)

    def test_sample(self):
        vars_size = 10
        dist = Binomial(params = np.array([0.0, 0.5, 1]))
        sample = dist.sample(1)
        self.assertTrue(\
            sample[0, 0] == 0.0 and \
            sample[0, 1] <= 1.0 and \
            sample[0, 1] >= 0.0 and \
            sample[0, 2] == 1.0)

if __name__ == '__main__':
    unittest.main()
