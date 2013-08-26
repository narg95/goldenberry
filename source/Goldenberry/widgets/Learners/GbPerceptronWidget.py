"""
<name>Perceptron</name>
<description>Canonical Perceptron Algorithm</description>
<contact>Nestor Rodriguez</contact>
<icon>icons/perceptron.png</icon>
<priority>50</priority>

"""
from Goldenberry.widgets import *

class GbPerceptronWidget(OWWidget):
    """Widget for the perceptron algorithm"""
    
    #attributes
    settingsList = ['learning_rate', 'max_iters', 'name']
    learning_rate = 1.0
    max_iters = 10
    learner = None
    classifier = None
    preprocessor = None
    data = None
    name = "Perceptron"

    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'Perceptron')
        
        self.setup_interfaces()
        self.setup_ui() 

    def setup_interfaces(self):
        self.inputs = [("Data", Orange.core.ExampleTable, self.setData),
                       ("Preprocess", PreprocessedLearner, self.setPreprocessor)]
        self.outputs = [("Learner", Orange.core.Learner),
                        ("Classifier", Orange.core.Classifier)]

    def setup_ui(self):
        # Loads the UI from an .ui file.
        self.controlArea = uic.loadUi(os.path.dirname(__file__) + "\\GbPerceptronWidget.ui", self)    
        
        # Subscribe to signals
        QObject.connect(self.buttonBox,QtCore.SIGNAL("accepted()"), self.accepted)
        QObject.connect(self.buttonBox,QtCore.SIGNAL("rejected()"), self.rejected)

        #set new binding controls
        nameEditor = OWGUI.lineEdit(self, self, "name", label="Name")
        learningEditor = OWGUI.lineEdit(self, self, "learning_rate", label="Learning Rage", valueType = float, validator = QDoubleValidator(0.0,1.0, 4, self.controlArea))
        maxiterEditor = OWGUI.lineEdit(self, self, "max_iters", label="Max. Iterations", valueType = int, validator = QIntValidator(1,10000, self.controlArea))
        
        self.paramBox.setLayout(QFormLayout(self.paramBox))
        self.paramBox.layout().addRow(nameEditor.box, nameEditor)
        self.paramBox.layout().addRow(learningEditor.box, learningEditor)
        self.paramBox.layout().addRow(maxiterEditor.box, maxiterEditor)
        
    def accepted(self):
        self.accept()
        self.apply_settings()

    def rejected(self):
        self.reject()

    def setData(self,data):
        self.data = self.isDataWithClass(data, Orange.core.VarTypes.Discrete, checkMissing=True) and data or None
        print "data has been set"
        self.apply_settings()

    def setPreprocessor(self, pp):
        self.preprocessor = pp

    def apply_settings(self):
        self.classifier = None
        self.learner = PerceptronLearner(self.max_iters, self.learning_rate)
        
        if self.preprocessor:
            self.learner = self.preprocessor.wrapLearner(self.learner)

        if None != self.data:
            self.classifier = self.learner(self.data)
        
        self.send("Learner", self.learner)
        self.send("Classifier", self.classifier)
        print "Learner and classifier was sent"

    def handleNewSignals(self):
        self.apply_settings()

if __name__=="__main__":
    test_widget()

def test_widget():
    appl = QApplication(sys.argv)
    ow = GbPerceptronWidget()
    ow.show()
    appl.exec_()    
    data = Orange.data.Table(os.path.dirname(__file__) + "\\test_data_2d.tab")
    ow.setData(data)
    ow.classifier(data[0])    