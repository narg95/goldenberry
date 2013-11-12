"""
<name>Kernel Perceptron</name>
<description>Kernel Perceptron Algorithm</description>
<contact>Nestor Rodriguez</contact>
<icon>icons/perceptron.svg</icon>
<priority>50</priority>

"""
from Goldenberry.widgets import *

class GbPerceptronWidget(OWWidget):
    """Widget for the perceptron algorithm"""
    
    #attributes
    settingsList = ['lr', 'max_iter', 'name']
    lr = 1.0
    max_iter = 10
    learner = None
    classifier = None
    preprocessor = None
    kernel = None
    data = None
    name = "Perceptron"

    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'Perceptron')
        
        self.setup_interfaces()
        self.setup_ui() 

    def setup_interfaces(self):
        self.inputs = [("Data", Orange.core.ExampleTable, self.set_data),
                       ("Kernel Function", GbKernel, self.set_kernel)]
        self.outputs = [("Learner", Orange.core.Learner),
                        ("Classifier", Orange.core.Classifier),
                        ("Learner Factory", GbFactory)]

    def setup_ui(self):
        # Loads the UI from an .ui file.
        self.name = self.captionTitle
        load_widget_ui(self)
        
        # Subscribe to signals
        QObject.connect(self.buttonBox,QtCore.SIGNAL("accepted()"), self.accepted)
        QObject.connect(self.buttonBox,QtCore.SIGNAL("rejected()"), self.rejected)

        #set new binding controls
        nameEditor = OWGUI.lineEdit(self, self, "name", label="Name")
        learningEditor = OWGUI.lineEdit(self, self, "lr", label="Learning Rage", valueType = float, validator = QDoubleValidator(0.0,1.0, 4, self.controlArea))
        maxiterEditor = OWGUI.lineEdit(self, self, "max_iter", label="Epochs", valueType = int, validator = QIntValidator(1,10000, self.controlArea))
        
        self.paramBox.setLayout(QFormLayout(self.paramBox))
        self.paramBox.layout().addRow(nameEditor.box, nameEditor)
        self.paramBox.layout().addRow(learningEditor.box, learningEditor)
        self.paramBox.layout().addRow(maxiterEditor.box, maxiterEditor)
        
    def accepted(self):
        self.accept()
        self.apply_settings()

    def rejected(self):
        self.reject()

    def set_data(self,data):
        self.data = self.isDataWithClass(data, Orange.core.VarTypes.Discrete, checkMissing=True) and data or None
        print "data has been set"
        self.apply_settings()

    def set_preprocessor(self, pp):
        self.preprocessor = pp

    def set_kernel(self, kernel):
        self.kernel = kernel(None) if None is not kernel else None

    def apply_settings(self):
        self.classifier = None
        self.learner = PerceptronLearner(self.kernel, self.max_iter, self.lr)        
        
        if self.preprocessor:
            self.learner = self.preprocessor.wrapLearner(self.learner)

        if None is not self.data:
            self.classifier = self.learner(self.data)

        parameters = {}
        for attr in ("kernel", "max_iter", "lr"):
            parameters[attr] = getattr(self, attr)
        
        self.send("Learner", self.learner)
        self.send("Classifier", self.classifier)
        self.send("Learner Factory", GbFactory(PerceptronLearner, parameters))
        
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
    ow.set_data(data)
    ow.classifier(data[0])    