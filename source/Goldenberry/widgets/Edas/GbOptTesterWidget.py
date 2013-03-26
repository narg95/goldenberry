"""
<name>Optimizer Tester</name>
<description>Evaluates the performance of a set of optimizers.</description>
<contact>Leidy Garzon</contact>
<icon>icons/TestOptim.png</icon>
<priority>200</priority>
"""

from Goldenberry.widgets import GbBaseOptimizer, uic, QtCore, GbOptimizersTester, QtGui, OWWidget, OWGUI, QTableWidget, Qt, Multiple
import thread

class GbOptTesterWidget(OWWidget):
    runs_results=[]
    experiment_results=[]
    optimizers=[]
    total_runs= 20

    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent,signalManager, title='Optimaze Tester')
        self.optimizers = []
        self.setup_interfaces()
        self.setup_ui()
        self.resize(755,435)
        
    def setup_interfaces(self):
         self.inputs = [("Optimizer", GbBaseOptimizer, self.set_optimizer, Multiple)]
         self.outputs = None
    
    def setup_ui(self):
        OWGUI.spin(self.controlArea, self, "total_runs", 1, 1000, box="Number of Runs")
        box = OWGUI.widgetBox(self.controlArea, "Run Number")
        OWGUI.button(self.controlArea, self, "Run", callback = self.execute)
        self.tabs = OWGUI.tabWidget(self.mainArea)
        self.experimentTab = OWGUI.createTabPage(self.tabs, "Experiment")
        self.runTab = OWGUI.createTabPage(self.tabs, "Especific Run")
        self.define_tables()
        self.adjustSize()

    def define_tables(self):        
        self.experiments_table = OWGUI.table(self.experimentTab, selectionMode=QTableWidget.MultiSelection)
        self.experiments_table.setColumnCount(13)
        self.experiments_table.setHorizontalHeaderLabels(["Name","Mean(Evals.)","Var(Evals.)","Min(Evals.)", "Max(Evals.)","Mean(Costs)","Var(Costs)","Min(Costs)", "Max(Costs)","Mean(Mean)","Var(Mean)","Min(Mean)", "Max(Mean)","Mean(Vars.)","Var(Vars.)","Min(Vars.)", "Max(Vars.)"])
        self.runs_table = OWGUI.table(self.runTab, selectionMode=QTableWidget.NoSelection)
        self.runs_table.setColumnCount(11)
        self.runs_table.setHorizontalHeaderLabels(["Name","#Run","Best","Cost"," evals", "found(min)", "found(max)", "min", "max", "mean", "variance"])

    def set_optimizer(self, optimizer, id=None):
         self.optimizers.append(optimizer)
        
    def execute(self):        
        self.experiment_results=[]
        self.runs_results = []
        self.experiments_table.setRowCount(0)
        self.runs_table.setRowCount(0)
        
        for idx , (optimizer, optimizer_name) in enumerate([(opt, name) for opt, name in self.optimizers if opt.ready()]):
            tester = GbOptimizersTester()
            run_results, test_results = tester.run(optimizer, self.total_runs)
            self.runs_results += [(optimizer_name,) + item for item in run_results]
            self.experiment_results.append((optimizer_name, ) + test_results)
            
        self.show_results(self.experiments_table, self.experiment_results)
        self.show_results(self.runs_table, self.runs_results)
        self.experiments_table.sortByColumn(3, Qt.AscendingOrder)            

    def show_results(self, table, results):
        total_runs = len(results)

        for rowidx, result_item in enumerate(results):
            table.insertRow(rowidx)
            for colidx, item in enumerate(result_item):
                table.setItem(rowidx, colidx, QtGui.QTableWidgetItem(str(item)))

if __name__=="__main__":
    test_widget()

def test_widget():
    app = QApplication(sys.argv)
    w = GbOptTesterWidget()
    w.show()
    app.exec_()