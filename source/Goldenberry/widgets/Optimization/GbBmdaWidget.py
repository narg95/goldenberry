"""
<name>Bmda</name>
<description>Bivariate marginal distribution of estimation algorithm</description>
<contact>Nestor Rodriguez</contact>
<icon>icons/Bmda.png</icon>
<priority>100</priority>

"""

from Goldenberry.widgets import *

class GbBmdaWidget(OWWidget):
    """Widget for Bmda algorithm"""
    
    #attributes
    settingsList = ['cand_size', 'var_size', 'max_evals']
    bmdaAlgorithm = Bmda()
    cand_size = 20
    var_size = 10
    max_evals = None
    cost_function = None

    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'Bmda')
        
        self.setup_interfaces()
        self.setup_ui() 

    def setup_interfaces(self):
        self.inputs = [("Cost Function", GbBaseCostFunction, self.set_cost_function)]
        self.outputs = [("Search Algorithm", GbBaseOptimizer)]

    def setup_ui(self):
        # Loads the UI from an .ui file.
        self.controlArea = uic.loadUi(os.path.dirname(__file__) + "\\GbBmdaWidget.ui", self)    
        
        # Subscribe to signals
        QObject.connect(self.applyButton,QtCore.SIGNAL("clicked()"), self.apply)
        QObject.connect(self.runButton,QtCore.SIGNAL("clicked()"), self.run)

        #set new binding controls
        popEditor = OWGUI.lineEdit(self, self, "cand_size", label="Population", valueType = int, validator = QIntValidator(4,10000, self.controlArea))
        varEditor = OWGUI.lineEdit(self, self, "var_size", label="Variables", valueType = int, validator = QIntValidator(4,10000, self.controlArea))
        maxEditor = OWGUI.lineEdit(self, self, "max_evals", label="Max Evaluations", valueType = int, validator = QIntValidator(0, 10000000, self.controlArea))
        self.paramBox.setLayout(QFormLayout(self.paramBox))
        self.paramBox.layout().addRow(varEditor.box, varEditor)
        self.paramBox.layout().addRow(popEditor.box, popEditor)
        self.paramBox.layout().addRow(maxEditor.box, maxEditor)

    def set_cost_function(self, cost_func):
        self.cost_function = cost_func
        self.apply()

    def apply(self):
        self.bmdaAlgorithm.setup(self.cost_function, self.var_size, self.cand_size, max_evals = self.max_evals)
        self.send("Search Algorithm" , self.bmdaAlgorithm)
        self.runButton.setEnabled(self.bmdaAlgorithm.ready())

    def run(self):
        if self.bmdaAlgorithm.ready():
            result = self.bmdaAlgorithm.search()
            self.resultTextEdit.setText("Evals: " + str(self.max_evals) + "\n"+ str(result))