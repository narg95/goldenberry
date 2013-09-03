from unittest import *
from Goldenberry.widgets import QtCore, Kernels
from Goldenberry.optimization.cost_functions import *
from Goldenberry.optimization.edas.Univariate import *
from Goldenberry.optimization.base.GbCostFunction import *
from PyQt4.QtGui import QApplication
from Optimization.GbCgaWidget import GbCgaWidget
from Optimization.GbBmdaWidget import GbBmdaWidget
from Optimization.GbUmdaWidget import GbUmdaWidget
from Optimization.GbPbilWidget import GbPbilWidget
from Optimization.GbCostFuncsWidget import GbCostFuncsWidget
from Optimization.GbTildaWidget import GbTildaWidget
from Optimization.GbBlackBoxWidget import GbBlackBoxWidget
from Learners.GbKernelBuilderWidget import GbKernelBuilderWidget

import sys

class WidgetsTest(TestCase):
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
        self.assertEqual(widget.functions.values()[widget.func_sel_index[0]], OneMax)
        #Uncomment only when testing the widget UI
        #widget.show()
        #self.app.exec_()     

    def test_cost_function_custom_code(self):        
        widget = GbCostFuncsWidget()
        widget.tabs.setCurrentIndex(1)
        widget.accepted()
        #Uncomment only when testing the widget UI
        #widget.show()
        #self.app.exec_()     

    def test_kernel_builder_basic(self):        
        widget = GbKernelBuilderWidget()
        custom_item = widget.funcs_listbox.findItems(Kernels.LinealKernel.__name__, QtCore.Qt.MatchExactly)[0]
        widget.funcs_listbox.setCurrentItem(custom_item)
        widget.accepted()
        self.assertEqual(widget.functions.values()[widget.func_sel_index[0]], Kernels.LinealKernel)
        #Uncomment only when testing the widget UI
        widget.show()
        self.app.exec_()     

    def test_kernel_builder_custom_code(self):        
        widget = GbKernelBuilderWidget()
        widget.tabs.setCurrentIndex(1)
        widget.accepted()
        #Uncomment only when testing the widget UI
        widget.show()
        self.app.exec_()     

    def test_test_optim_base(self):        
        var_size, cand_size = 10, 20
        widget = GbBlackBoxWidget()
        optimizer1 = Cga()
        optimizer1.cost_func = GbCostFunction(OneMax)
        optimizer1.setup(var_size, cand_size)
        widget.set_optimizer((optimizer1, 'Cga'),0)

        optimizer2 = Tilda()
        optimizer2.cost_func = GbCostFunction(OneMax)
        optimizer2.setup(var_size, cand_size)
        widget.set_optimizer((optimizer2, 'Tilda'),1)

        widget.execute()
        self.assertEqual(widget.total_runs * 2, len(widget.runs_results))
        self.assertEqual(widget.experiments_table.columnCount(),12)
        self.assertEqual(widget.experiments_table.rowCount(),2)
        self.assertEqual(widget.runs_table.columnCount(),12)
        self.assertEqual(widget.runs_table.rowCount(),20 * 2)
        #Uncomment only when testing the widget UI
        #widget.show()
        #self.app.exec_()

    def tearDown(self):
       pass
