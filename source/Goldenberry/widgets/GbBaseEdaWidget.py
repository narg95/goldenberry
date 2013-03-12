import abc
import os
from Goldenberry.widgets import OWWidget, QObject, QtCore, QFormLayout, OWGUI, QIntValidator, GbBaseCostFunction, GbBaseOptimizer, load_widget_ui

class GbBaseEdaWidget(OWWidget):
    """Widget for the base Eda algorithm"""
    
    def __init__(self, parent=None, signalManager=None, title = "base"):
        OWWidget.__init__(self, parent, signalManager, title)
        
        self.setup_interfaces()
        self.setup_ui() 

    #attributes
    settingsList = ['cand_size', 'var_size', 'max_evals']
    optimizer = None
    cand_size = 20
    var_size = 10
    max_evals = None
    cost_function = None

    def setup_interfaces(self):
        pass


    def setup_ui(self):

        load_widget_ui(self)

        # Subscribe to signals
        QObject.connect(self.applyButton,QtCore.SIGNAL("clicked()"), self.apply)
        QObject.connect(self.runButton,QtCore.SIGNAL("clicked()"), self.run)

        #set new binding controls
        popEditor = OWGUI.lineEdit(self, self, "cand_size", label="# Candidates", valueType = int, validator = QIntValidator(4,10000, self.controlArea))
        varEditor = OWGUI.lineEdit(self, self, "var_size", label="Variables", valueType = int, validator = QIntValidator(4,10000, self.controlArea))
        maxEditor = OWGUI.lineEdit(self, self, "max_evals", label="Max Evals.", valueType = int, validator = QIntValidator(0, 100000, self.controlArea))
        self.paramBox.setLayout(QFormLayout(self.paramBox))
        self.paramBox.layout().addRow(varEditor.box, varEditor)
        self.paramBox.layout().addRow(popEditor.box, popEditor)
        self.paramBox.layout().addRow(maxEditor.box, maxEditor)
        self.runButton.setEnabled(False)

    def setup_interfaces(self):
        self.inputs = [("Cost Function", GbBaseCostFunction, self.set_cost_function)]
        self.outputs = [("Optimizer", GbBaseOptimizer)]
        
    def set_cost_function(self, cost_func):
        self.optimizer.cost_func = None
        if None != cost_func:
            func_type, args = cost_func
            self.optimizer.cost_func = func_type(*args)
            self.runButton.setEnabled(self.optimizer.ready())

    def setup_optimizer(self):
        self.optimizer.setup(self.var_size, self.cand_size, max_evals = self.max_evals)

    def apply(self): 
        self.setup_optimizer()       
        self.send("Optimizer" , self.optimizer)
        self.runButton.setEnabled(self.optimizer.ready())

    def run(self):
        self.optimizer.reset()
        if self.optimizer.ready():
            result = self.optimizer.search()
            evals, argmin, argmax, min, max, mean, stdev = self.optimizer.cost_func.statistics()
            self.resultTextEdit.setText("Best: %s\ncost:%s\n#evals:%s\n#argmin:%s\nargmax:%s\nmin val:%s\nmax val:%s\nmean:%s\nstdev:%s"%(result.params, result.cost, evals, argmin, argmax, min, max, mean, stdev))