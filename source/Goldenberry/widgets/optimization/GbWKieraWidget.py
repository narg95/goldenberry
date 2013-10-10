"""
<name>WKiera Cost Function</name>
<description>This is a cost function to facilitate the creation of a WKiera algorithm.</description>
<contact>Nestor Rodriguez</contact>
<icon>icons/Pbil.svg</icon>
<priority>200</priority>
"""

from Goldenberry.widgets import OWWidget, GbCostFunction, WKieraCostFunction, QFormLayout, OWGUI, load_widget_ui
from Orange.core import Learner
from orange import ExampleTable
class GbWKieraWidget(OWWidget):
    """WKiera cost function widget."""
    data = None
    learner = None
    var_size = None
    weight = 1

    settingsList = ['weight', 'var_size']

    def __init__(self, parent=None, signalManager=None, title="WKiera Cost Function"):
        OWWidget.__init__(self, parent, signalManager, title)
        self.setup_interfaces()
        self.setup_ui()

    def setup_ui(self):
        load_widget_ui(self)
        weight_control = OWGUI.hSlider(self, self, 'weight', minValue = 0, maxValue = 1000, step = 1, divideFactor = 1000.0, labelFormat = "%.3f", label = "Weight")
        self.varEdit = OWGUI.lineEdit(self, self, "var_size", label="Variables")
        applyButton = OWGUI.button(self, self, label = "Apply", callback = self.apply)
        self.varEdit.setEnabled(False)
        self.paramslayout.addRow(self.varEdit.box, self.varEdit)
        self.paramslayout.addRow(weight_control.box, weight_control)
        self.paramslayout.addRow(applyButton)

    def setup_interfaces(self):
        self.outputs = [("Cost Function", GbCostFunction)]
        self.inputs = [("Learner", Learner, self.set_learner ),
                       ("Data", ExampleTable, self.set_data)]        

    def set_learner(self, learner):
        self.learner = learner
        self.apply()

    def set_data(self, data):
        self.data = data
        self.varEdit.setText("" if data is None else str(len(data.domain.attributes)))
        self.apply()

    def apply(self):
        if self.learner is not None and self.data is not None:
            wkiera_cost_func = lambda _: WKieraCostFunction(self.data, self.learner, solution_weight = self.weight/1000.0)
            self.send("Cost Function", wkiera_cost_func)
            
    