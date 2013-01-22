from Goldenberry.optimization.edas import Cga, Bmda
from Goldenberry.optimization.cost_functions.functions import *
from unittest import *

class CgaTest(TestCase):
    """Test class for the Cga algorithm"""
    def test_basic(self):
        cga = Cga.Cga()
        cga.setup(onemax(), 10, 20)
        result = cga.search()
        self.assertTrue(result.params.all())
        self.assertEqual(result.cost, 10)

class BmdaTest(TestCase):
    """Test class for the Bmda algorithm"""
    def test_basic(self):
        bmda = Bmda.Bmda()
        bmda.setup(onemax(), 10, 20)
        result = bmda.search()
        self.assertTrue(result.params.all())
        self.assertEqual(result.cost, 10)

if __name__ == '__main__':
    unittest.main()
