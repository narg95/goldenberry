"""
<name>cGA</name>
<description>Compact Genetic Algorithm</description>
<contact>Nestor Rodriguez</contact>
<icon>icons/Cga.png</icon>
<priority>100</priority>

"""

from Goldenberry.widgets import *

class GbCgaWidget(OWWidget):
    """Widget for cga algorithm"""
    
    #attributes
    settingsList = ['popsize', 'varsize', 'maxgens']
    cgaAlgorithm = Cga()
    popsize = 20
    varsize = 10
    maxgens = None
    cost_function = None

    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'cGA')
        
        self.setup_interfaces()
        self.setup_ui() 

    def setup_interfaces(self):
        self.inputs = [("Cost Function", GbBaseCostFunction, self.set_cost_function)]
        self.outputs = [("Search Algorithm", GbBaseOptimizer)]

    def setup_ui(self):
        # Loads the UI from an .ui file.
        self.controlArea = uic.loadUi(os.path.dirname(__file__) + "\\GbCgaWidget.ui", self)    
        
        # Subscribe to signals
        QObject.connect(self.applyButton,QtCore.SIGNAL("clicked()"), self.apply)
        QObject.connect(self.runButton,QtCore.SIGNAL("clicked()"), self.run)

        #set new binding controls
        popEditor = OWGUI.lineEdit(self, self, "popsize", label="Population", valueType = int, validator = QIntValidator(4,10000, self.controlArea))
        varEditor = OWGUI.lineEdit(self, self, "varsize", label="Variables", valueType = int, validator = QIntValidator(4,10000, self.controlArea))
        maxEditor = OWGUI.lineEdit(self, self, "maxgens", label="Max Epochs", valueType = int, validator = QIntValidator(0, 100000, self.controlArea))
        self.paramBox.setLayout(QFormLayout(self.paramBox))
        self.paramBox.layout().addRow(varEditor.box, varEditor)
        self.paramBox.layout().addRow(popEditor.box, popEditor)
        self.paramBox.layout().addRow(maxEditor.box, maxEditor)

    def set_cost_function(self, cost_func):
        self.cost_function = cost_func
        self.apply()

    def apply(self):
        self.cgaAlgorithm.setup(self.cost_function, self.varsize, self.popsize, self.maxgens)
        self.send("Search Algorithm" , self.cgaAlgorithm )
        self.runButton.setEnabled(self.cgaAlgorithm.ready())

    def run(self):
        if self.cgaAlgorithm.ready():
            result = self.cgaAlgorithm.search()
            self.resultTextEdit.setText(str(result))
