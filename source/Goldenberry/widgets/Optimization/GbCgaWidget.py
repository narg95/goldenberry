"""
<name>cGA</name>
<description>Compact Genetic Algorithm</description>
<contact>Nestor Rodriguez</contact>
<icon>icons/Cga.svg</icon>
<priority>10</priority>

"""

from Goldenberry.widgets.GbBaseEdaWidget import GbBaseEdaWidget
from Goldenberry.widgets import Cga, GbCostFunction, GbBaseOptimizer

class GbCgaWidget(GbBaseEdaWidget):
    """Widget for cga algorithm"""
    
    def __init__(self, parent=None, signalManager=None):
        self.optimizer = Cga()
        GbBaseEdaWidget.__init__(self, parent, signalManager, 'cGA')
        self.inputs = [("Cost Function", GbCostFunction, self.set_cost_function)]
        self.outputs = [("Optimizer", GbBaseOptimizer)]
            