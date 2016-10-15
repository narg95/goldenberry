from Goldenberry.widgets import OWWidget, inspect, GbDynamicFunction, load_widget_ui, OWGUI, QObject, QApplication, QClipboard, QtCore

class GbDynamicFunctionWidget(OWWidget):
    """Provides a dynamic function builder widget."""
    
    settingsList = ['func_sel_index']
    functions = {}
    func_sel_index = []

    def __init__(self, func_module , parent=None, signalManager=None, title = 'Function Builder', output_channel = "Function"):
        OWWidget.__init__(self, parent, signalManager, title)
        self.output_channel = output_channel
        self.setup_ui()
        self.setup_interfaces()
        self.functions = dict(inspect.getmembers(func_module, lambda member: inspect.isfunction(member)))
    
    def setup_interfaces(self):
        raise NotImplementedError()

    def setup_ui(self):
        """Configures the user interface"""
        # Loads the UI from an .ui file.
        load_widget_ui(self)
        
        #set up the ui controls
        self.funcs_listbox = OWGUI.listBox(self.groupBox, self, "func_sel_index", "functions", callback = self.func_selected)
        if len(self.functions):
            self.funcs_listbox.setCurrentRow(0)

        # Subscribe to signals
        QObject.connect(self.buttonBox,QtCore.SIGNAL("accepted()"), self.accepted)
        QObject.connect(self.buttonBox,QtCore.SIGNAL("rejected()"), self.rejected)
        QObject.connect(self.copy_button,QtCore.SIGNAL("clicked()"), self.copy)
        QObject.connect(self.paste_button,QtCore.SIGNAL("clicked()"), self.paste)
    
    def copy(self):
        QApplication.clipboard().setText(self.benchmark_text.toPlainText(), QClipboard.Clipboard)
    
    def paste(self):
        self.custom_text.setText(QApplication.clipboard().text())

    def func_selected(self):
        self.benchmark_text.clear()
        if len(self.func_sel_index) > 0:
            func_type = self.functions.values()[self.func_sel_index[0]]
            source_text = inspect.getsource(func_type)
            self.benchmark_text.setText(source_text)

    def accepted(self):
        if self.tabs.currentWidget() == self.benchmark_tab:
            if len(self.func_sel_index) > 0:
                func = self.functions.values()[self.func_sel_index[0]]
                self.accept()
                self.send(self.output_channel, lambda _: self.create_function(func = func))        
        else: 
            if len(self.custom_text.toPlainText())  > 0:
                func_text = str(self.custom_text.toPlainText()).strip()
                func = self.create_function(script = func_text)
                self.accept()
                self.send(self.output_channel, lambda _: self.create_function(script = func_text))

    def create_function(self, func = None, script = None):
        raise NotImplementedError()
                
    def rejected(self):
        self.reject()

if __name__=="__main__":
    test_widget()

def test_widget():
    app = QApplication(sys.argv)
    w = GbCostFuncsWidget()
    w.show()
    app.exec_()