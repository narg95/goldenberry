from Goldenberry.optimization.edas.Cga import Cga
from Goldenberry.optimization.cost_functions.functions import *
import unittest

class CgaTest(unittest.TestCase):
    def test_basic(self):
        cga = Cga()
        cga.setup(onemax(), 10, 20)
        cga.search()

if __name__ == '__main__':
    unittest.main()
