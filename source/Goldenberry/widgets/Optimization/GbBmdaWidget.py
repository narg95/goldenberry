"""
<name>Bmda</name>
<description>Compact Genetic Algorithm</description>
<contact>Nestor Rodriguez</contact>
<icon>icons/Eda.png</icon>
<priority>100</priority>

"""

from Goldenberry.widgets.GbBaseEdaWidget import GbBaseEdaWidget
from Goldenberry.widgets import Bmda, GbBaseCostFunction, GbBaseOptimizer

class GbBmdaWidget(GbBaseEdaWidget):
    """Widget for Bmda algorithm"""
    
    def __init__(self, parent=None, signalManager=None):
        self.optimizer = Bmda()
        GbBaseEdaWidget.__init__(self, parent, signalManager, 'Bmda')
        self.inputs = [("Cost Function", GbBaseCostFunction, self.set_cost_function)]
        self.outputs = [("Optimizer", GbBaseOptimizer)]