from unittest import *
from Goldenberry.optimization.cost_functions.functions import *
from PyQt4.QtGui import QApplication
from Goldenberry.widgets.optimization.GbCgaWidget import GbCgaWidget
import sys

class OptimizationWidgetsTest(TestCase):
    """Test the widgets for the optimization part."""

    def setUp(self):
        self.app = QApplication(sys.argv)

    def test_cga_basic(self):        
        widget = GbCgaWidget()
        widget.apply()
        
        self.assertFalse(widget.runButton.isEnabled())
        widget.set_cost_function(onemax())

        self.assertTrue(widget.runButton.isEnabled())
        widget.run()        

    def tearDown(self):
       pass