from Goldenberry.optimization.edas.Cga import Cga
from Goldenberry.optimization.cost_functions.functions import *
from unittest import *

class CgaTest(TestCase):
    """Test class for the Cga algorithm"""
    def test_basic(self):
        cga = Cga()
        cga.setup(onemax(), 10, 20)
        result = cga.search()
        self.assertTrue(result.params.all())
        self.assertEqual(result.cost, 10)

if __name__ == '__main__':
    unittest.main()
