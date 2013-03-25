"""
<name>Cost Functions</name>
<description>Provide a set of fitness functions</description>
<contact>Nestor Rodriguez</contact>
<icon>icons/Cost.png</icon>
<priority>200</priority>
"""

from Goldenberry.widgets import *

class GbCostFuncsWidget(OWWidget):
    """Widget for fitness functions"""
    
    settingsList = ['cost_func_sel_index']
    cost_funcs = dict(inspect.getmembers(cost_functions, lambda member: inspect.isfunction(member)))
    cost_func_sel_index = []

    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'fitness')
        
        self.setup_interfaces()
        self.setup_ui()
    
    def setup_interfaces(self):
        self.outputs = [("Cost Function", GbCostFunction)]

    def setup_ui(self):
        """Configures the user interface"""
        # Loads the UI from an .ui file.
        self.controlArea = uic.loadUi(os.path.dirname(__file__) + "\\GbCostFuncsWidget.ui", self)
        
        #set up the ui controls
        self.funcs_listbox = OWGUI.listBox(self.groupBox, self, "cost_func_sel_index", "cost_funcs", box="Cost Functions", callback = self.func_selected)
        if len(self.cost_funcs):
            self.funcs_listbox.setCurrentRow(0)

        # Subscribe to signals
        QObject.connect(self.buttonBox,QtCore.SIGNAL("accepted()"), self.accepted)
        QObject.connect(self.buttonBox,QtCore.SIGNAL("rejected()"), self.rejected)
    
    def func_selected(self):
        if len(self.cost_func_sel_index) > 0:
            func_type = self.cost_funcs.values()[self.cost_func_sel_index[0]]
            source_text = self.get_formated_source(func_type)
            self.customText.clear()
            self.customText.setText(source_text)

    def get_formated_source(self, func_type):
        return "".join([text[4:] if text.startswith("    ") else text for text in inspect.getsourcelines(func_type)[0][1:] ])

    def accepted(self):
        if len(self.cost_func_sel_index) > 0:
            func = self.cost_funcs.values()[self.cost_func_sel_index[0]]
            self.accept()
            self.send("Cost Function" , (func, None))
        
        elif len(self.customText.toPlainText().strip()) > 0:
            func_text = str(self.customText.toPlainText())
            cost_func = GbCostFunction(script = func_text)
            self.accept()
            self.send("Cost Function", (None, func_text))

    def rejected(self):
        self.reject()

if __name__=="__main__":
    test_widget()

def test_widget():
    app = QApplication(sys.argv)
    w = GbCostFuncsWidget()
    w.show()
    app.exec_()