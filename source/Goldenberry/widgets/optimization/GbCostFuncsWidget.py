from Optimization import *

"""
<name>Cost Functions</name>
<description>Provide a set of fitness functions</description>
<contact>Nestor Rodriguez</contact>
<icon>icons/Cost.png</icon>
<priority>200</priority>
"""

class GbCostFuncsWidget(OWWidget):
    """Widget for fitness functions"""
    
    settingsList = ['cost_func_sel_index']
    cost_funcs = dict(inspect.getmembers(functions, lambda member: inspect.isclass(member) and not inspect.isabstract(member) ))
    cost_func_sel_index = []

    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'fitness')
        
        self.setup_interfaces()
        self.setup_ui()
    
    def setup_interfaces(self):
        self.outputs = [("Cost Functions", GbBaseCostFunction)]

    def setup_ui(self):
        """Configures the user interface"""
        # Loads the UI from an .ui file.
        self.controlArea = uic.loadUi(os.path.dirname(__file__) + "\\GbCostFuncsWidget.ui", self)
        
        #set up the ui controls
        OWGUI.listBox(self.groupBox, self, "cost_func_sel_index", "cost_funcs", box="Cost Functions")

        # Subscribe to signals
        QObject.connect(self.buttonBox,QtCore.SIGNAL("accepted()"), self.accepted)
        QObject.connect(self.buttonBox,QtCore.SIGNAL("rejected()"), self.rejected)
        
    def accepted(self):
        self.accept()
        cost_func = self.cost_funcs[self.cost_funcs.keys()[self.cost_func_sel_index[0]]]
        self.send("Cost Functions" , cost_func)

    def rejected(self):
        self.reject()

if __name__=="__main__":
    test_widget()

def test_widget():
    app = QApplication(sys.argv)
    w = GbCostFuncsWidget()
    w.show()
    app.exec_()