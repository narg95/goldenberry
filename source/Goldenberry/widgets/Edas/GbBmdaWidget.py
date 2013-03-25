"""
<name>Bmda</name>
<description>Bivariate marginal distribution algorithm.</description>
<contact>Nestor Rodriguez</contact>
<icon>icons/Eda.png</icon>
<priority>100</priority>

"""

from Goldenberry.widgets.GbBaseEdaWidget import GbBaseEdaWidget
from Goldenberry.widgets import Bmda, GbCostFunction, GbBaseOptimizer

class GbBmdaWidget(GbBaseEdaWidget):
    """Widget for Bmda algorithm"""
    
    def __init__(self, parent=None, signalManager=None):
        self.optimizer = Bmda()
        GbBaseEdaWidget.__init__(self, parent, signalManager, 'Bmda')
        self.inputs = [("Cost Function", GbCostFunction, self.set_cost_function)]
        self.outputs = [("Optimizer", GbBaseOptimizer)]