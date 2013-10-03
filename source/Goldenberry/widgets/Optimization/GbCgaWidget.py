"""
<name>cGA</name>
<description>Compact Genetic Algorithm</description>
<contact>Nestor Rodriguez</contact>
<icon>icons/Cga.svg</icon>
<priority>10</priority>

"""

from Goldenberry.widgets.optimization.GbBaseEdaWidget import GbBaseEdaWidget
from Goldenberry.widgets import Cga, GbCostFunction, GbBaseOptimizer

class GbCgaWidget(GbBaseEdaWidget):
    """Widget for cga algorithm"""
    
    def __init__(self, parent=None, signalManager=None, title = "Cga"):
        self.optimizer = Cga()
        super(GbCgaWidget, self).__init__(parent, signalManager, title)
        self.inputs = [("Cost Function", GbCostFunction, self.set_cost_function)]
        self.outputs = [("Optimizer", GbBaseOptimizer)]