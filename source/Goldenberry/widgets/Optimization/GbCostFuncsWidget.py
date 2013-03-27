"""
<name>Cost Function</name>
<description>Provide a set of fitness functions</description>
<contact>Nestor Rodriguez</contact>
<icon>icons/Cost.svg</icon>
<priority>1010</priority>
"""

from Goldenberry.widgets import *

class GbCostFuncsWidget(OWWidget):
    """Provides a cost function."""
    
    settingsList = ['cost_func_sel_index']
    cost_funcs = dict(inspect.getmembers(cost_functions, lambda member: inspect.isfunction(member)))
    cost_func_sel_index = []

    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'Cost Function')
        
        self.setup_interfaces()
        self.setup_ui()
    
    def setup_interfaces(self):
        self.outputs = [("Cost Function", GbCostFunction)]

    def setup_ui(self):
        """Configures the user interface"""
        # Loads the UI from an .ui file.
        self.controlArea = uic.loadUi(os.path.dirname(__file__) + "\\GbCostFuncsWidget.ui", self)
        
        #set up the ui controls
        self.funcs_listbox = OWGUI.listBox(self.groupBox, self, "cost_func_sel_index", "cost_funcs", callback = self.func_selected)
        if len(self.cost_funcs):
            self.funcs_listbox.setCurrentRow(0)

        # Subscribe to signals
        QObject.connect(self.buttonBox,QtCore.SIGNAL("accepted()"), self.accepted)
        QObject.connect(self.buttonBox,QtCore.SIGNAL("rejected()"), self.rejected)
    
    def func_selected(self):
        self.benchmark_text.clear()
        if len(self.cost_func_sel_index) > 0:
            func_type = self.cost_funcs.values()[self.cost_func_sel_index[0]]
            source_text = inspect.getsource(func_type)
            self.benchmark_text.setText(source_text)

    def accepted(self):
        if self.tabs.currentWidget() == self.benchmark_tab:
            if len(self.cost_func_sel_index) > 0:
                func = self.cost_funcs.values()[self.cost_func_sel_index[0]]
                self.accept()
                self.send("Cost Function" , (func, None))        
        else: 
            if len(self.custom_text.toPlainText())  > 0:
                func_text = str(self.custom_text.toPlainText()).strip()

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