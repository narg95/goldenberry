import unittest
from optimization.edas.CgaTest import CgaTest

if __name__ == "__main__":

    loader = unittest.TestLoader()
    suite = unittest.TestSuite((
        loader.loadTestsFromTestCase(CgaTest),
        
        ))

    runner = unittest.TextTestRunner(verbosity = 2)
    runner.run(suite)