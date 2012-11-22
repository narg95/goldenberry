"""
<name>cGA</name>
<description>Compact Genetic Algorithm</description>
<contact>Nestor Rodriguez</contact>
<icon>icons/cga.png</icon>
<priority>100</priority>

"""

from Goldenberry.optimization.edas.Cga import Cga

class GbCgaWidget(OWWidget):
    """Widget for cga algorithm"""
    
    #attributes
    settingsList = ['popsize', 'varsize', 'maxgens']
    cgaAlgorithm = Cga()
    popsize = 20
    varsize = 10
    maxgens = None
    fitness = None

    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'cGA')
        self.inputs = [("Fitness Function", baseFitness, self.setFitness)]
        self.outputs = [("Search Algorithm", baseSearcher)]
        self.setupUi() 

    def setupUi(self):
        # Loads the UI from an .ui file.
        self.controlArea = uic.loadUi(os.path.dirname(__file__) + "\\GbCgaWidget.ui", self)    
        
        # Subscribe to signals
        QObject.connect(self.buttonBox,QtCore.SIGNAL("accepted()"), self.accepted)
        QObject.connect(self.buttonBox,QtCore.SIGNAL("rejected()"), self.rejected)

        #set new binding controls
        popEditor = OWGUI.lineEdit(self, self, "popsize", label="Population size", validator = QIntValidator(2,100, self.controlArea))
        varEditor = OWGUI.lineEdit(self, self, "varsize", label="Variables size", validator = QIntValidator(2,100, self.controlArea))
        maxEditor = OWGUI.lineEdit(self, self, "maxgens", label="Max Generations", validator = QIntValidator(0, 10000, self.controlArea))
        self.paramBox.setLayout(QFormLayout(self.paramBox))
        self.paramBox.layout().addRow(varEditor.box, varEditor)
        self.paramBox.layout().addRow(popEditor.box, popEditor)
        self.paramBox.layout().addRow(maxEditor.box, maxEditor)

    def setFitness(self, fitness):
        self.fitness = fitness

    def accepted(self):
        self.accept()
        self.cgaAlgorithm.config(self.fitness, self.varsize, self.popsize, self.maxgens)
        self.send(INTERFACES.SEARCH_ALGORITHM[0] , self.cgaAlgorithm )

    def rejected(self):
        self.reject()

if __name__=="__main__":
    testWidget()

def testWidget():
    appl = QApplication(sys.argv)
    ow = CGAWidget()
    #ow.fitness = onemax()
    ow.show()
    appl.exec_()
    #result = ow.cgaAlgorithm.find()
    #print(result)