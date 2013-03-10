"""
<name>Umda</name>
<description>Compact Genetic Algorithm</description>
<contact>Nestor Rodriguez</contact>
<icon>icons/Eda.png</icon>
<priority>100</priority>

"""

from Goldenberry.widgets.GbBaseEdaWidget import GbBaseEdaWidget
from Goldenberry.widgets import Pbil, GbBaseCostFunction, GbBaseOptimizer

class GbUmdaWidget(GbBaseEdaWidget):
    """Widget for umda algorithm"""
    
    def __init__(self, parent=None, signalManager=None):
        self.optimizer = Pbil()
        GbBaseEdaWidget.__init__(self, parent, signalManager, 'Umda')
        self.inputs = [("Cost Function", GbBaseCostFunction, self.set_cost_function)]
        self.outputs = [("Optimizer", GbBaseOptimizer)]
    
    def setup_optimizer(self):
        self.optimizer.setup(self.var_size, self.cand_size, max_evals = self.max_evals, learning_rate = 1.0)   