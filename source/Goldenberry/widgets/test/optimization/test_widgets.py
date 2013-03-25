from unittest import *
from Goldenberry.widgets import QtCore
from Goldenberry.optimization.cost_functions import *
from Goldenberry.optimization.base.GbCostFunction import *
from PyQt4.QtGui import QApplication
from Edas.GbCgaWidget import GbCgaWidget
from Edas.GbBmdaWidget import GbBmdaWidget
from Edas.GbUmdaWidget import GbUmdaWidget
from Edas.GbPbilWidget import GbPbilWidget
from Edas.GbCostFuncsWidget import GbCostFuncsWidget
from Edas.GbTildaWidget import GbTildaWidget

import sys

class OptimizationWidgetsTest(TestCase):
    """Test the widgets for the optimization part."""

    def setUp(self):
        self.app = QApplication(sys.argv)

    def test_cga_basic(self):        
        widget = GbCgaWidget()
        widget.apply()
        
        self.assertFalse(widget.runButton.isEnabled())
        widget.set_cost_function((OneMax, None))

        self.assertTrue(widget.runButton.isEnabled())
        widget.run()   

    def test_umda_basic(self):        
        widget = GbUmdaWidget()
        widget.apply()
        
        self.assertFalse(widget.runButton.isEnabled())
        widget.set_cost_function((OneMax, None))

        self.assertTrue(widget.runButton.isEnabled())
        widget.run()   
        
        #Uncomment only when testing the widget UI
        #widget.show()
        #self.app.exec_()  
    
    def test_tilda_basic(self):        
        widget = GbTildaWidget()
        widget.apply()
        
        self.assertFalse(widget.runButton.isEnabled())
        widget.set_cost_function((OneMax, None))

        self.assertTrue(widget.runButton.isEnabled())
        widget.run()   
        
        #Uncomment only when testing the widget UI
        #widget.show()
        #self.app.exec_()      
            
    def test_pbil_basic(self):        
        widget = GbPbilWidget()
        widget.apply()
        
        self.assertFalse(widget.runButton.isEnabled())
        widget.set_cost_function((OneMax, None))

        self.assertTrue(widget.runButton.isEnabled())
        widget.run()   
        
        #Uncomment only when testing the widget UI
        #widget.show()
        #self.app.exec_()     
        
    def test_bmda_basic(self):        
        widget = GbBmdaWidget()
        widget.apply()
        
        self.assertFalse(widget.runButton.isEnabled())
        widget.set_cost_function((OneMax, None))

        self.assertTrue(widget.runButton.isEnabled())
        widget.run()   
        
        #Uncomment only when testing the widget UI
        #widget.show()
        #self.app.exec_()
        
    def test_cost_function_basic(self):        
        widget = GbCostFuncsWidget()
        custom_item = widget.funcs_listbox.findItems(OneMax.__name__, QtCore.Qt.MatchExactly)[0]
        widget.funcs_listbox.setCurrentItem(custom_item)
        widget.accepted()
        self.assertEqual(widget.cost_funcs.values()[widget.cost_func_sel_index[0]], OneMax)
        #Uncomment only when testing the widget UI
        #widget.show()
        #self.app.exec_()     

    def test_cost_function_custom_code(self):        
        widget = GbCostFuncsWidget()
        zeromax = widget.funcs_listbox.findItems(ZeroMax.__name__, QtCore.Qt.MatchExactly)[0]
        widget.funcs_listbox.setCurrentItem(zeromax)
        widget.accepted()
        self.assertEqual(widget.cost_funcs.values()[widget.cost_func_sel_index[0]], ZeroMax)
        #Uncomment only when testing the widget UI
        #widget.show()
        #self.app.exec_()     

    def tearDown(self):
       pass