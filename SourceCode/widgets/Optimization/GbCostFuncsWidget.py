from Optimization import *

"""
<name>Fitness</name>
<description>Provide a set of fitness functions</description>
<contact>Nestor Rodriguez</contact>
<icon>icons/cGA.png</icon>
<priority>200</priority>
"""

class GbCostFuncsWidget(OWWidget):
    """Widget for fitness functions"""
    
    settingsList = ['selected_cost_func']
    cost_funcs = dict(inspect.getmembers(functions, lambda member: inspect.isclass(member) and not inspect.isabstract(member) ))
    selected_cost_func = []

    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'fitness')
        
        #Sets the provided and required interfaces
        self.outputs = [("Cost Functions", GbBaseCostFunction)]
        self.setupUi()

    def setupUi(self):
        """Configures the user interface"""
        # Loads the UI from an .ui file.
        self.controlArea = uic.loadUi(os.path.dirname(__file__) + "\\GbCostFuncsWidget.ui", self)
        
        #set up the ui controls
        OWGUI.listBox(self.groupBox, self, "selected_cost_func", "cost_funcs",
              box="Cost Functions")

        # Subscribe to signals
        QObject.connect(self.buttonBox,QtCore.SIGNAL("accepted()"), self.accepted)
        
    def accepted(self):
        self.accept()
        self.send("Cost Functions" , self.selected_cost_func )

if __name__=="__main__":
    test_widget()

def test_widget():
    app = QApplication(sys.argv)
    w = GbCostFuncsWidget()
    w.show()
    app.exec_()