"""
<name>Black Box Tester</name>
<description>Evaluates the performance of a set of optimizers.</description>
<contact>Leidy Garzon</contact>
<icon>icons/Blackbox.svg</icon>
<priority>1020</priority>
"""

from Goldenberry.widgets import GbBaseOptimizer, uic, QtCore, GbBlackBoxTester, QtGui, OWWidget, OWGUI, QTableWidget, Qt, Multiple
import thread

class GbBlackBoxWidget(OWWidget):
    runs_results=[]
    experiment_results=[]
    optimizers={}
    total_runs= 20

    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent,signalManager, title='Optimaze Tester')
        self.setup_interfaces()
        self.setup_ui()
        self.resize(755,435)
        
    def setup_interfaces(self):
         self.inputs = [("Optimizer", GbBaseOptimizer, self.set_optimizer, Multiple)]
    
    def setup_ui(self):
        OWGUI.spin(self.controlArea, self, "total_runs", 1, 1000, box="Number of Runs")
        OWGUI.button(self.controlArea, self, "Run", callback = self.execute)
        self.tabs = OWGUI.tabWidget(self.mainArea)
        self.experimentTab = OWGUI.createTabPage(self.tabs, "Experiment")
        self.runTab = OWGUI.createTabPage(self.tabs, "Especific Run")
        self.define_tables()
        OWGUI.button(self.mainArea, self, "Copy Tables to Clipboard", callback = self.copy_table, tooltip ='Copy the information of the two tables to the clipboard and paste It')
        self.adjustSize()

    def define_tables(self):        
        self.experiments_table = OWGUI.table(self.experimentTab, selectionMode=QTableWidget.MultiSelection)
        self.experiments_table.setColumnCount(18)
        self.experiments_table.setHorizontalHeaderLabels(["Name","Mean(Evals.)","Var(Evals.)","Min(Evals.)", "Max(Evals.)","Mean(Costs)","Var(Costs)","Min(Costs)", "Max(Costs)","Mean(Mean)","Var(Mean)","Min(Mean)", "Max(Mean)","Mean(Vars.)","Var(Vars.)","Min(Vars.)", "Max(Vars.)", "Total Time"])
        self.experiments_table.setSortingEnabled(True)
        self.runs_table = OWGUI.table(self.runTab, selectionMode=QTableWidget.MultiSelection)
        self.runs_table.setColumnCount(12)
        self.runs_table.setHorizontalHeaderLabels(["Name","#Run","Best","Cost"," evals", "found(min)", "found(max)", "min", "max", "mean", "variance", "time"])
        self.runs_table.setSortingEnabled(True)

    def set_optimizer(self, optimizer, id=None):
        if self.optimizers.has_key(id):
            del self.optimizers[id]
        
        if None != optimizer:
            self.optimizers[id] = optimizer
        
    def execute(self):        
        self.experiment_results=[]
        self.runs_results = []
        self.experiments_table.setRowCount(0)
        self.runs_table.setRowCount(0)
        
        for idx , (optimizer, optimizer_name) in enumerate([(opt, name) for opt, name in self.optimizers.values() if opt.ready()]):
            tester = GbBlackBoxTester()
            run_results, test_results = tester.test(optimizer, self.total_runs)
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
        r = Tk()
        r.withdraw()
        r.clipboard_clear()
        r.clipboard_append(text)
        r.destroy()

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

if __name__=="__main__":
    test_widget()

def test_widget():
    app = QApplication(sys.argv)
    w = GbBlackBoxWidget()
    w.show()
    app.exec_()
