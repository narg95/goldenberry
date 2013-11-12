"""
<name>Wrapper Cost Function</name>
<description>This is a cost function to create wrapper features selection methods.</description>
<contact>Nestor Rodriguez</contact>
<icon>icons/Pbil.svg</icon>
<priority>200</priority>
"""

from Goldenberry.widgets import OWWidget, GbCostFunction, GbWrapperCostFunction, QFormLayout, OWGUI, load_widget_ui, GbFactory
from orange import ExampleTable
class GbWKieraWidget(OWWidget):
    """Wrapper cost function widget."""
    data = None
    factory = None
    var_size = None
    weight = 1
    folds = 10
    normalization = 1

    settingsList = ['weight', 'var_size', 'folds', 'normalization']

    def __init__(self, parent=None, signalManager=None, title="WKiera Cost Function"):
        OWWidget.__init__(self, parent, signalManager, title)
        self.setup_interfaces()
        self.setup_ui()

    def setup_ui(self):
        load_widget_ui(self)
        weight_control = OWGUI.hSlider(self, self, 'weight', minValue = 0, maxValue = 1000, step = 1, divideFactor = 1000.0, labelFormat = "%.3f", label = "Accuracy/Subset-size tradeoff")
        self.varEdit = OWGUI.lineEdit(self, self, "var_size", label="Variables", valueType = int)
        foldsEdit = OWGUI.hSlider(self, self, 'folds', minValue = 2, maxValue = 20, step = 1, label = "Folds")
        applyButton = OWGUI.button(self, self, label = "Apply", callback = self.apply)
        normalizCheck = OWGUI.checkBox(self, self, "normalization", label="Normalize data", tooltip="Use data normalization")
        self.varEdit.setEnabled(False)
        self.paramslayout.addRow(self.varEdit.box, self.varEdit)
        self.paramslayout.addRow(weight_control.box, weight_control)
        self.paramslayout.addRow(foldsEdit.box, foldsEdit)
        self.paramslayout.addRow(normalizCheck)
        self.paramslayout.addRow(applyButton)

    def setup_interfaces(self):
        self.outputs = [("Cost Function", GbCostFunction)]
        self.inputs = [("Learner Factory", GbFactory, self.set_factory ),
                       ("Data", ExampleTable, self.set_data)]        

    def set_factory(self, factory):
        self.factory = factory
        self.apply()

    def set_data(self, data):
        self.data = data
        self.varEdit.setText("" if data is None else str(len(data.domain.attributes)))
        self.apply()

    def apply(self):
        if self.factory is not None and self.data is not None:
            wkiera_cost_func = lambda _: GbWrapperCostFunction(self.data, self.factory, solution_weight = (self.weight/1000.0), folds = self.folds, normalization = self.normalization)
            self.send("Cost Function", wkiera_cost_func)
            
    