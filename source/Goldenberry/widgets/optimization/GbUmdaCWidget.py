"""
<name>UMDAc</name>
<description>Continuous Univariate Marginal Distribution Algorithm.</description>
<contact>Nestor Rodriguez</contact>
<icon>icons/Pbilc.svg</icon>
<priority>60</priority>

"""

from Goldenberry.widgets.optimization.GbBaseEdaWidget import GbBaseEdaWidget
from Goldenberry.widgets import Pbilc, GbCostFunction, GbBaseOptimizer, QDoubleValidator, OWGUI, GbSolution

class GbUmdaCWidget(GbBaseEdaWidget):
    """Widget for UMDAc algorithm."""
    
    learning_rate = 1.0

    def __init__(self, parent=None, signalManager=None):
        self.optimizer = Pbilc()
        self.settingsList.append("learning_rate")
        GbBaseEdaWidget.__init__(self, parent, signalManager, 'UMDAc')
        self.inputs = [("Cost Function", GbCostFunction, self.set_cost_function)]
        self.outputs = [("Optimizer", GbBaseOptimizer), ("Solution", GbSolution)]
    
    def setup_ui(self):
        GbBaseEdaWidget.setup_ui(self)

    def setup_optimizer(self):
        self.optimizer.setup(self.cand_size, max_evals = self.max_evals, learning_rate = 1.0)   