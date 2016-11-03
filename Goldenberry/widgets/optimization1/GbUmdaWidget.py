"""
<name>UMDA</name>
<description>Univariate marginal distribution algorithm.</description>
<contact>Nestor Rodriguez</contact>
<icon>icons/Pbil.svg</icon>
<priority>30</priority>
"""
from Goldenberry.widgets.optimization.GbBaseEdaWidget import GbBaseEdaWidget
from Goldenberry.widgets import Pbil, GbCostFunction, GbBaseOptimizer, GbSolution

class GbUmdaWidget(GbBaseEdaWidget):
    """Widget for umda algorithm"""
    
    def __init__(self, parent=None, signalManager=None, title = None):
        self.optimizer = Pbil()
        super(GbUmdaWidget, self).__init__(parent, signalManager, 'UMDA')
        self.inputs = [("Cost Function", GbCostFunction, self.set_cost_function)]
        self.outputs = [("Optimizer", GbBaseOptimizer), ("Solution", GbSolution)]
        
    def setup_optimizer(self):
        self.optimizer.setup(self.cand_size, max_evals = self.max_evals, learning_rate = 1.0)   