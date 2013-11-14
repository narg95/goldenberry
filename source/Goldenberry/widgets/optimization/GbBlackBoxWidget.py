"""
<name>Black Box Tester</name>
<description>Evaluates the performance of a set of optimizers.</description>
<contact>Leidy Garzon</contact>
<icon>icons/Blackbox.svg</icon>
<priority>1020</priority>
"""

from Goldenberry.widgets import *

class GbBlackBoxWidget(OWWidget):
    runs_results=[]
    experiment_results=[]
    optimizers={}
    total_runs= 20

    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent,signalManager, title='Optimaze Tester')
        self.setup_interfaces()
        self.setup_ui()
        
    def setup_interfaces(self):
         self.inputs = [("Optimizer", GbBaseOptimizer, self.set_optimizer, Multiple)]
         self.outputs = [("Solution", GbSolution)]
    
    def setup_ui(self):
        load_widget_ui(self)
        self.clean_layout()
        self.layout().addWidget(self.main_widget)
        OWGUI.spin(self.paramBox, self, "total_runs", 1, 1000)
        self.define_tables()
        OWGUI.button(self.details_tab, self, "Send", self.send_candidate)

        # Subscribe to signals
        QObject.connect(self.run_button,QtCore.SIGNAL("clicked()"), self.execute)
        #QObject.connect(self.stop_button,QtCore.SIGNAL("clicked()"), self.stop)
        QObject.connect(self.copy_button,QtCore.SIGNAL("clicked()"), self.copy_table)

    def clean_layout(self):
        while(self.layout().count() > 0):
            self.layout().removeItem(self.layout().takeAt(0))

    def define_tables(self):        
        self.experiments_table = OWGUI.table(self.summary_tab, selectionMode=QTableWidget.SingleSelection)
        self.experiments_table.setColumnCount(12)
        self.experiments_table.setHorizontalHeaderLabels(["Name",\
            "Cost(max)", "Cost(avg)", "#Evals(avg)", "Time[s](avg)", \
            "Cost(std)", "#Evals(std)", "Time[s](std)", \
            "Cost(min)", "#Evals(min)", "Time[s](min)", \
            "#Evals(max)", "Time[s](max)"])
        self.experiments_table.resizeColumnsToContents()

        self.runs_table = OWGUI.table(self.details_tab, selectionMode=QTableWidget.SingleSelection)
        self.runs_table.setColumnCount(12)
        self.runs_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.runs_table.setHorizontalHeaderLabels([\
            "Name","Best","Cost","#Evals", "Time[s]",\
            "Avg", "Std", "Min", "Max", "Min(idx.)", "Max(idx.)", "#Run", "Roots", "Children"])
        #Disabled until the current Item can be obtained when the grid has been sorted. See method send_candidate
        #self.runs_table.setSortingEnabled(True)
        self.runs_table.resizeColumnsToContents()

    def set_optimizer(self, optimizer, id=None):
        if self.optimizers.has_key(id):
            del self.optimizers[id]
        
        if None is not optimizer:
            self.optimizers[id] = optimizer

    #def stop(self):
    #    self.test.terminate()
    #    self.disconnect(self.test, QtCore.SIGNAL('progress(PyQt_PyObject)'), self.progress)
    #    self.run_button.setEnabled(True)
    #    self.stop_button.setEnabled(False)
            
    def execute(self):
        self.progressBarInit()
        self.run_button.setEnabled(False)
        #self.stop_button.setEnabled(True)
        self.progressBarSet(0)
        self.test = Search(self)
        self.connect(self.test, QtCore.SIGNAL('progress(PyQt_PyObject)'), self.update_progress)
        self.test.start()

    def run_tests(self, callback = None):        
        self.experiment_results=[]
        self.runs_results = []
        self.candidates = []
        self.experiments_table.setRowCount(0)
        self.runs_table.setRowCount(0)        

        optimizers = [(opt, name) for opt, name in self.optimizers.values() if opt.ready()]
        for idx , (optimizer, optimizer_name) in enumerate(optimizers):
            tester = GbBlackBoxTester()
            run_results, test_results, candidates = tester.test(optimizer, self.total_runs, lambda best, progress: callback(best, (progress  + idx)/float(len(optimizers))))
            self.candidates += candidates
            self.runs_results += [(optimizer_name,) + item for item in run_results]
            self.experiment_results.append((optimizer_name, ) + test_results)
            
        self.show_results(self.experiments_table, self.experiment_results)
        self.show_results(self.runs_table, self.runs_results)
        self.experiments_table.sortByColumn(3, Qt.AscendingOrder)    
        
        testcase=self.experiments_table.rowCount()       
        testcasecol=self.experiments_table.columnCount()  

    def show_results(self, table, results):
        total_runs = len(results)

        for rowidx, result_item in enumerate(results):
            table.insertRow(rowidx)
            for colidx, item in enumerate(result_item):
                table.setItem(rowidx, colidx, QtGui.QTableWidgetItem(str(item)))            

    def copy_table(self):
        text=''
        text=text + self.selectTableItems(self.experiments_table)
        text=text + self.selectTableItems(self.runs_table)
        QApplication.clipboard().setText(text, QClipboard.Clipboard)

    def selectTableItems(self, table):
        num_rows=0
        num_cols=0
        num_rows, num_cols = table.rowCount(), table.columnCount()
        #text=''
        text=self.get_table_header(table)
        for row in range(num_rows):
            rows = []
            for col in range(num_cols):
                item = table.item(row, col)
                text = text + '\t'+item.text()
                rows.append(item.text() if item else '')
            text=text+'\n'
        return text + '\n'

    def get_table_header(self,table):
        num_cols=0
        num_cols = table.columnCount()
        text=''
        rows = []
        for col in range(num_cols):
                item = table.horizontalHeaderItem(col)
                text = text + '\t'+item.text()
                rows.append(item.text() if item else '')
        return text + '\n'

    def send_candidate(self):
        if self.runs_table.currentRow() >=0 :
            self.send("Solution", self.candidates[self.runs_table.currentRow()])

    def update_progress(self, progressArgs):
        progress = int(progressArgs.progress * 100)
        self.progressBarSet(progress)
        if progress >= 100:
            self.progressBarFinished()
            self.run_button.setEnabled(True)
            #self.stop_button.setEnabled(False)

class Search(QtCore.QThread, QObject):
    
    def __init__(self, widget):    
        QtCore.QThread.__init__(self)       
        self.widget = widget

    progress = pyqtSignal(object)
     
    def run(self):
        self.widget.run_tests(callback = self.current_progress)

    def current_progress(self, result, progress):
        self.progress.emit(ProgressArgs(result, progress))

class ProgressArgs:
    def __init__(self, result, progress):
        self.result = result
        self.progress = progress

if __name__=="__main__":
    test_widget()

def test_widget():
    app = QApplication(sys.argv)
    w = GbBlackBoxWidget()
    w.show()
    app.exec_()
