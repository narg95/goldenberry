import unittest
from Goldenberry.optimization.unit_test.CgaTest import CgaTest

if __name__ == "__main__":

    loader = unittest.TestLoader()
    suite = unittest.TestSuite((
        loader.loadTestsFromTestCase(CgaTest),
        
        ))

    runner = unittest.TextTestRunner(verbosity = 2)
    runner.run(suite)