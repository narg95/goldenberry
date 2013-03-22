"""
<name>cGA</name>
<description>Compact Genetic Algorithm</description>
<contact>Nestor Rodriguez</contact>
<icon>icons/Eda.png</icon>
<priority>100</priority>

"""

from Goldenberry.widgets.GbBaseEdaWidget import GbBaseEdaWidget
from Goldenberry.widgets import Cga, GbBaseCostFunction, GbBaseOptimizer

class GbCgaWidget(GbBaseEdaWidget):
    """Widget for cga algorithm"""
    
    def __init__(self, parent=None, signalManager=None):
        self.optimizer = Cga()
        GbBaseEdaWidget.__init__(self, parent, signalManager, 'cGA')
        self.inputs = [("Cost Function", GbBaseCostFunction, self.set_cost_function)]
        self.outputs = [("Optimizer", GbBaseOptimizer)]
            