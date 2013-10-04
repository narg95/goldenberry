"""
<name>PBILc</name>
<description>Continuous Population-Based incremental learning algorithm.</description>
<contact>Nestor Rodriguez</contact>
<icon>icons/Pbilc.svg</icon>
<priority>50</priority>

"""

from Goldenberry.widgets.optimization.GbBaseEdaWidget import GbBaseEdaWidget
from Goldenberry.widgets import Pbilc, GbCostFunction, GbBaseOptimizer, QDoubleValidator, OWGUI, GbSolution

class GbPbilCWidget(GbBaseEdaWidget):
    """Widget for PBILc algorithm."""
    
    learning_rate = 1.0

    def __init__(self, parent=None, signalManager=None):
        self.optimizer = Pbilc()
        self.settingsList.append("learning_rate")
        GbBaseEdaWidget.__init__(self, parent, signalManager, 'PBILc')
        self.inputs = [("Cost Function", GbCostFunction, self.set_cost_function)]
        self.outputs = [("Optimizer", GbBaseOptimizer), ("Solution", GbSolution)]
    
    def setup_ui(self):
        GbBaseEdaWidget.setup_ui(self)
        learning_editor = OWGUI.lineEdit(self, self, "learning_rate", label="Learning Rate", valueType = float, validator = QDoubleValidator(0.0, 1.0, 4, self.controlArea))
        self.paramBox.layout().addRow(learning_editor.box, learning_editor)

    def setup_optimizer(self):
        self.optimizer.setup(self.var_size, self.cand_size, max_evals = self.max_evals, learning_rate = self.learning_rate)   