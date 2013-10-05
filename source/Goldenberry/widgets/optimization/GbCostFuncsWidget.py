"""
<name>Cost Function</name>
<description>Provide a set of fitness functions</description>
<contact>Nestor Rodriguez</contact>
<icon>icons/Cost.svg</icon>
<priority>1010</priority>
"""

from Goldenberry.widgets import *
from Goldenberry.widgets.GbDynamicFunctionWidget import GbDynamicFunctionWidget
from Goldenberry.widgets import cost_functions

class GbCostFuncsWidget(GbDynamicFunctionWidget):
    """Provides a cost function."""
   
    def __init__(self, parent=None, signalManager=None):
        self.var_size = 10
        super(GbCostFuncsWidget, self).__init__(cost_functions, parent, signalManager, 'Cost Function', "Cost Function")

    def setup_interfaces(self):
        self.outputs = [("Cost Function", GbCostFunction)]

    def setup_ui(self):
        super(GbCostFuncsWidget, self).setup_ui()
        self.varsizewidget.layout().addWidget(OWGUI.lineEdit(self, self, "var_size", label="Number of Variables", valueType = int, validator = QIntValidator(2,1000000, self.controlArea)))

    def create_function(self, func = None, script = None):
        return GbCostFunction(func= func, script = script, var_size = self.var_size)

if __name__=="__main__":
    test_widget()

def test_widget():
    app = QApplication(sys.argv)
    w = GbCostFuncsWidget()
    w.show()
    app.exec_()