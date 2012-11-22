"""
<name>cGA</name>
<description>Compact Genetic Algorithm</description>
<contact>Nestor Rodriguez</contact>
<icon>icons/cga.png</icon>
<priority>100</priority>

"""
from Optimization import *

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
        self.inputs = [("Fitness Function", GbBaseCostFunction, self.cost_function)]
        self.outputs = [("Search Algorithm", GbBaseOptimizer)]
        self.setupUi() 

    def setupUi(self):
        # Loads the UI from an .ui file.
        self.controlArea = uic.loadUi(os.path.dirname(__file__) + "\\GbCgaWidget.ui", self)    
        
        # Subscribe to signals
        QObject.connect(self.buttonBox,QtCore.SIGNAL("accepted()"), self.accepted)
        QObject.connect(self.buttonBox,QtCore.SIGNAL("rejected()"), self.rejected)

        #set new binding controls
        popEditor = OWGUI.lineEdit(self, self, "popsize", label="Population size", valueType = int, validator = QIntValidator(4,10000, self.controlArea))
        varEditor = OWGUI.lineEdit(self, self, "varsize", label="Variables size", valueType = int, validator = QIntValidator(4,10000, self.controlArea))
        maxEditor = OWGUI.lineEdit(self, self, "maxgens", label="Max Generations", valueType = int, validator = QIntValidator(0, 100000, self.controlArea))
        self.paramBox.setLayout(QFormLayout(self.paramBox))
        self.paramBox.layout().addRow(varEditor.box, varEditor)
        self.paramBox.layout().addRow(popEditor.box, popEditor)
        self.paramBox.layout().addRow(maxEditor.box, maxEditor)

    def setFitness(self, fitness):
        self.fitness = fitness

    def accepted(self):
        self.accept()
        self.cgaAlgorithm.setup(self.cost_function, self.varsize, self.popsize, self.maxgens)
        self.send("Search Algorithm" , self.cgaAlgorithm )

    def rejected(self):
        self.reject()

if __name__=="__main__":
    testWidget()

def testWidget():
    appl = QApplication(sys.argv)
    ow = GbCgaWidget()
    ow.cost_function = onemax()
    ow.show()
    appl.exec_()
    result = ow.cgaAlgorithm.search()
    print(result.params)