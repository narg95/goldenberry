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
        super(GbCostFuncsWidget, self).__init__(cost_functions, GbCostFunction, parent, signalManager, 'Cost Function', "Cost Function")

    def setup_interfaces(self):
        self.outputs = [("Cost Function", GbCostFunction)]

if __name__=="__main__":
    test_widget()

def test_widget():
    app = QApplication(sys.argv)
    w = GbCostFuncsWidget()
    w.show()
    app.exec_()