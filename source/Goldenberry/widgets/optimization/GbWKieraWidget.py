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
    test_data = None
    factory = None
    var_size = None
    weight = 100
    folds = 10
    test_folds = 10
    normalization = 1
    title = "Wrapper Cost Function"

    settingsList = ['weight', 'var_size', 'folds', 'normalization']

    def __init__(self, parent=None, signalManager=None, title="Wrapper Cost Function"):
        OWWidget.__init__(self, parent, signalManager, title)
        self.setup_interfaces()
        self.setup_ui()

    def setup_ui(self):
        load_widget_ui(self)
        nameEditor = OWGUI.lineEdit(self, self, "title", label="Title", callback=self.change_title)
        weight_control = OWGUI.hSlider(self, self, 'weight', minValue = 0, maxValue = 1000, step = 1, divideFactor = 1000.0, labelFormat = "%.3f", label = "Accuracy/Subset-size tradeoff")
        self.varEdit = OWGUI.lineEdit(self, self, "var_size", label="Variables", valueType = int)
        foldsEdit = OWGUI.hSlider(self, self, 'folds', minValue = 2, maxValue = 20, step = 1, label = "Training Folds")
        testfoldsEdit = OWGUI.hSlider(self, self, 'test_folds', minValue = 2, maxValue = 20, step = 1, label = "Test Folds")
        applyButton = OWGUI.button(self, self, label = "Apply", callback = self.apply)
        normalizCheck = OWGUI.checkBox(self, self, "normalization", label="Normalize data", tooltip="Use data normalization")
        self.varEdit.setEnabled(False)
        self.paramslayout.addRow(nameEditor.box, nameEditor)
        self.paramslayout.addRow(self.varEdit.box, self.varEdit)
        self.paramslayout.addRow(weight_control.box, weight_control)
        self.paramslayout.addRow(foldsEdit.box, foldsEdit)
        self.paramslayout.addRow(testfoldsEdit.box, testfoldsEdit)
        self.paramslayout.addRow(normalizCheck)
        self.paramslayout.addRow(applyButton)

    def change_title(self):
        self.setCaption(self.title)
        self.setWindowTitle(self.title)

    def setup_interfaces(self):
        self.outputs = [("Cost Function", GbCostFunction)]
        self.inputs = [("Learner Factory", GbFactory, self.set_factory ),
                       ("Training Data", ExampleTable, self.set_data),
                       ("Test Data", ExampleTable, self.set_test_data)]

    def set_factory(self, factory):
        self.factory = factory
        self.apply()

    def set_data(self, data):
        self.data = data
        self.varEdit.setText("" if data is None else str(len(data.domain.attributes)))
        self.apply()

    def set_test_data(self, test_data):
        self.test_data = test_data
        self.varEdit.setText("" if test_data is None else str(len(test_data.domain.attributes)))
        self.apply()

    def apply(self):
        if self.factory is not None and self.data is not None and self.test_data is not None:
            wkiera_cost_func = lambda _: GbWrapperCostFunction(self.data, self.test_data, self.factory, solution_weight = (self.weight/1000.0), folds = self.folds, test_folds= self.test_folds, normalization = self.normalization)
            self.send("Cost Function", wkiera_cost_func)
            
    