"""
<name>UMDAc</name>
<description>Continuous Univariate Marginal Distribution Algorithm.</description>
<contact>Nestor Rodriguez</contact>
<icon>icons/Pbilc.svg</icon>
<priority>60</priority>

"""

from Goldenberry.widgets.GbBaseEdaWidget import GbBaseEdaWidget
from Goldenberry.widgets import Tilda, GbCostFunction, GbBaseOptimizer, QDoubleValidator, OWGUI

class GbUmdaCWidget(GbBaseEdaWidget):
    """Widget for UMDAc algorithm."""
    
    learning_rate = 1.0

    def __init__(self, parent=None, signalManager=None):
        self.optimizer = Tilda()
        self.settingsList.append("learning_rate")
        GbBaseEdaWidget.__init__(self, parent, signalManager, 'UMDAc')
        self.inputs = [("Cost Function", GbCostFunction, self.set_cost_function)]
        self.outputs = [("Optimizer", GbBaseOptimizer)]
    
    def setup_ui(self):
        GbBaseEdaWidget.setup_ui(self)
        self.paramBox.layout().addRow(learning_editor.box, learning_editor)

    def setup_optimizer(self):
        self.optimizer.setup(self.var_size, self.cand_size, max_evals = self.max_evals, learning_rate = 1.0)   