"""
<name>BMDA</name>
<description>Bivariate marginal distribution algorithm.</description>
<contact>Nestor Rodriguez</contact>
<icon>icons/Bmda.svg</icon>
<priority>80</priority>

"""

from Goldenberry.widgets.optimization.GbBaseEdaWidget import GbBaseEdaWidget
from Goldenberry.widgets import Bmda, GbCostFunction, GbBaseOptimizer, OWGUI, Qt, DependencyMethod

class GbBmdaWidget(GbBaseEdaWidget):
    """Widget for Bmda algorithm"""
    
    def __init__(self, parent=None, signalManager=None):
        self.method = 0
        self.optimizer = Bmda()
        GbBaseEdaWidget.__init__(self, parent, signalManager, 'BMDA')
        self.inputs = [("Cost Function", GbCostFunction, self.set_cost_function)]
        self.outputs = [("Optimizer", GbBaseOptimizer)]

        #UI Buttons
        radio_box = OWGUI.radioButtonsInBox(self, self, "method",
              box = "Dependency Method",
              btnLabels = ["Chi Square", "SIM"])
        self.verticalLayoutWidget.layout().addWidget(radio_box)

    def setup_optimizer(self):
        self.optimizer.setup(self.var_size, self.cand_size, max_evals = self.max_evals, dependency_method = DependencyMethod.chi2_test if self.method == 0 else DependencyMethod.sim)