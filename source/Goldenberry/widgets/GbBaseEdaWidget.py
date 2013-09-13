import abc
import os
import thread
from Goldenberry.widgets import OWWidget, QObject, QtCore, QFormLayout, OWGUI, QIntValidator, GbCostFunction, GbBaseOptimizer, load_widget_ui

class GbBaseEdaWidget(OWWidget):
    """Widget for the base Eda algorithm"""
    
    def __init__(self, parent=None, signalManager=None, title = "base"):
        OWWidget.__init__(self, parent, signalManager, title)
        
        self.setup_interfaces()
        self.setup_ui() 

    #attributes
    settingsList = ['cand_size', 'var_size', 'max_evals', 'name']
    optimizer = None
    cand_size = 20
    var_size = 10
    max_evals = 100
    cost_function = None
    name = None

    def setup_interfaces(self):
        pass

    def setup_ui(self):

        self.name = self.captionTitle
        load_widget_ui(self)

        # Subscribe to signals
        QObject.connect(self.applyButton,QtCore.SIGNAL("clicked()"), self.apply)
        QObject.connect(self.runButton,QtCore.SIGNAL("clicked()"), self.run)

        #set new binding controls
        nameEditor = OWGUI.lineEdit(self, self, "name", label="Name")
        popEditor = OWGUI.lineEdit(self, self, "cand_size", label="# Candidates", valueType = int, validator = QIntValidator(4,1000000, self.controlArea))
        varEditor = OWGUI.lineEdit(self, self, "var_size", label="Variables", valueType = int, validator = QIntValidator(4,1000000, self.controlArea))
        maxEditor = OWGUI.lineEdit(self, self, "max_evals", label="Max Evals.", valueType = int, validator = QIntValidator(0, 1000000, self.controlArea))
        self.paramBox.setLayout(QFormLayout(self.paramBox))
        self.paramBox.layout().addRow(nameEditor.box, nameEditor)
        self.paramBox.layout().addRow(varEditor.box, varEditor)
        self.paramBox.layout().addRow(popEditor.box, popEditor)
        self.paramBox.layout().addRow(maxEditor.box, maxEditor)
        self.runButton.setEnabled(False)
        self.progressBarInit()

    def setup_interfaces(self):
        self.inputs = [("Cost Function", GbCostFunction, self.set_cost_function)]
        self.outputs = [("Optimizer", GbBaseOptimizer)]
        
    def set_cost_function(self, func):
        self.optimizer.cost_func = None
        if None is not func:
            if type(func) is tuple:
                self.optimizer.cost_func = GbCostFunction(*func)
            elif type(func) is GbCostFunction or issubclass(type(func), GbCostFunction):
                func.reset_statistics()
                self.optimizer.cost_func = func
            self.runButton.setEnabled(self.optimizer.ready())

    def setup_optimizer(self):
        self.optimizer.setup(self.var_size, self.cand_size, max_evals = self.max_evals, callback_func = self.progress)

    def apply(self): 
        self.setup_optimizer()       
        self.send("Optimizer" , (self.optimizer, self.name))
        self.runButton.setEnabled(self.optimizer.ready())
    
    def progress(self, progress):
        if progress == 1.0:
            self.progressBarFinished()
        else:
            self.progressBarSet(int(progress * 100))

    def run(self):
        self.optimizer.reset()
        if self.optimizer.ready():
            thread.start_new_thread(self.search,())
            
    def search(self):
        result = self.optimizer.search()
        evals, argmin, argmax, min, max, mean, stdev = self.optimizer.cost_func.statistics()
        self.resultTextEdit.setText("Best: %s\ncost:%s\n#evals:%s\n#argmin:%s\nargmax:%s\nmin val:%s\nmax val:%s\nmean:%s\nstdev:%s"%(result.params, result.cost, evals, argmin, argmax, min, max, mean, stdev))