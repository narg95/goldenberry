import unittest
import os
import sys
import inspect
import imp

if __name__ == "__main__":
    test_modules = ['test.test_widgets.WidgetsTest']    
    suite = unittest.TestLoader().loadTestsFromNames(test_modules)    
    runner = unittest.TextTestRunner(verbosity = 2)
    runner.run(suite)

    

    