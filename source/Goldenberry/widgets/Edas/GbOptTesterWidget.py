"""
<name>Optimizer Tester</name>
<description>Evaluates the performance of a set of optimizers.</description>
<contact>Leidy Garzon</contact>
<icon>icons/TestOptim.png</icon>
<priority>200</priority>
"""

from Goldenberry.widgets import GbBaseOptimizer, uic, QtCore, GbOptimizersTester, QtGui, OWWidget, OWGUI, QTableWidget, Qt
import thread

class GbOptTesterWidget(OWWidget):
    indExp=1
    indexRun = 1
    tab =None
    num_evals = 20
    tester = None
    listNumberRuns=[]
    listRun=[]
    listExperiment=[]
    listOp=[]
    numberRuns=0

    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent,signalManager, title='Optimaze Tester')
        self.listOp=[]
        self.setup_interfaces()
        self.setup_ui()
        self.resize(755,435)
        
    def setup_interfaces(self):
         self.inputs = [("Optimizer", GbBaseOptimizer, self.set_optimizer, 0)]
         self.outputs = None
    
    def set_optimizer(self, optimizer, id=None):
         #self.optimizer = optimizer
         self.listOp.append(optimizer)
         #self.populateTableRun(self.tester.listParam)
         
         #print self.tester.listRun
         #self.tester.setup(self.optimizer,5)
         #if self.optimizer.ready():
             #self.tester.evaluate()
    def populateTableRun(self, listParam):
        i=0
        while i < len(listParam) :
            name,params, cost,evals, argmin, argmax, min, max, mean, stdev = listParam[i].split('-')
            self.tableRun.setRowCount(self.indexRun)
            self.tableRun.setItem(self.indexRun-1,0,QtGui.QTableWidgetItem(str(name)))
            self.tableRun.setItem(self.indexRun-1,1,QtGui.QTableWidgetItem(str(params)))
            self.tableRun.setItem(self.indexRun-1,2,QtGui.QTableWidgetItem(str(cost)))
            self.tableRun.setItem(self.indexRun-1,3,QtGui.QTableWidgetItem(str(evals)))
            self.tableRun.setItem(self.indexRun-1,4,QtGui.QTableWidgetItem(str(argmax)))
            self.tableRun.setItem(self.indexRun-1,5,QtGui.QTableWidgetItem(str(argmin)))
            self.tableRun.setItem(self.indexRun-1,6,QtGui.QTableWidgetItem(str(min)))
            self.tableRun.setItem(self.indexRun-1,7,QtGui.QTableWidgetItem(str(max)))
            self.tableRun.setItem(self.indexRun-1,8,QtGui.QTableWidgetItem(str(mean)))
            self.tableRun.setItem(self.indexRun-1,9,QtGui.QTableWidgetItem(str(stdev)))
            self.tableRun.setItem(self.indexRun-1,10,QtGui.QTableWidgetItem(str(i+1)))
            self.indexRun+=1
            i+=1
        self.tableExp.sortByColumn(2, Qt.AscendingOrder)

    def populateTableExp(self, listCals):
        
        while self.indExp-1 < len(self.listExperiment) :
            self.tableExp.setRowCount(self.indExp)
            self.tableExp.setItem(self.indExp-1,0,QtGui.QTableWidgetItem(str(listCals[9])))
            self.tableExp.setItem(self.indExp-1,1,QtGui.QTableWidgetItem(str(listCals[0])))
            self.tableExp.setItem(self.indExp-1,2,QtGui.QTableWidgetItem(str(listCals[1])))
            self.tableExp.setItem(self.indExp-1,3,QtGui.QTableWidgetItem(str(listCals[2][0])))
            self.tableExp.setItem(self.indExp-1,4,QtGui.QTableWidgetItem(str(listCals[2][1])))
            self.tableExp.setItem(self.indExp-1,5,QtGui.QTableWidgetItem(str(listCals[3])))
            self.tableExp.setItem(self.indExp-1,6,QtGui.QTableWidgetItem(str(listCals[4])))
            self.tableExp.setItem(self.indExp-1,7,QtGui.QTableWidgetItem(str(listCals[5][0])))
            self.tableExp.setItem(self.indExp-1,8,QtGui.QTableWidgetItem(str(listCals[5][1])))
            self.tableExp.setItem(self.indExp-1,9,QtGui.QTableWidgetItem(str(listCals[6])))
            self.tableExp.setItem(self.indExp-1,10,QtGui.QTableWidgetItem(str(listCals[7])))
            self.tableExp.setItem(self.indExp-1,11,QtGui.QTableWidgetItem(str(listCals[8][0])))
            self.tableExp.setItem(self.indExp-1,12,QtGui.QTableWidgetItem(str(listCals[8][1])))
            self.indExp+=1

    def execute(self):
        
        self.updateRunNumberBox()
        self.listExperiment=[]
        if self.tableExp.rowCount()!=0:
            j=0
            while j<self.tableExp.rowCount():
                self.tableExp.removeRow(j)
        if self.tableRun.rowCount()!=0:
            k=0
            while k<self.tableRun.rowCount():
                self.tableRun.removeRow(k)
        #self.tableRun = None
        self.indExp=1
        self.indexRun=1
        i=0

        while i<len(self.listOp):
            if(None!=self.listOp[i]):
                if(self.listOp[i].ready()):
                    self.tester = GbOptimizersTester()
                    self.tester.resetList()
                    thread.start_new_thread(self.tester.run,(self.listOp[i],self.numberRuns))
                    while(True):
                        if(self.tester.hasFinished):
                           self.listRun.append(self.tester.listParam)
                           self.listExperiment.append(self.tester.listCals)
                           self.populateTableExp(self.tester.listCals)
                           self.populateTableRun(self.tester.listParam)
                           break
            i+=1

    def updateRunNumberBox(self):
        self.runNumberBox.setEnabled(1)
        self.runNumberBox.clear()
        for i in range(1,self.numberRuns+1):
            self.runNumberBox.addItem(str(i))
        self.runNumberBox.addItem("Max")
        self.runNumberBox.addItem("Min")
             
    def setup_ui(self):
        self.chosenRun = 1
        self.numberRuns = 20
       #**********************************Control Area**************************************
        #Number of runs
        OWGUI.spin(self.controlArea, self, "numberRuns", 1, 1000, box="Number of Runs")
        
        #List of the number of runs
        box = OWGUI.widgetBox(self.controlArea, "Run Number")
        self.runNumberBox =OWGUI.comboBox(box, self, "chosenRun", label="Run to Detaild: ", items=[], tooltip='Select the run to detaild', callback=self.setTableRunDetail)
        self.runNumberBox.setEnabled(0)
        self.adjustSize()
   
        #Accept Button
        OWGUI.button(self.controlArea, self, "Run", callback = self.execute)
        #table
        
        #Results Tabs
        self.tabs = OWGUI.tabWidget(self.mainArea)
        self.experimentTab = OWGUI.createTabPage(self.tabs, "Experiment")
        self.runTab = OWGUI.createTabPage(self.tabs, "Especific Run")
        self.createTable()

        #Table Example
    def createTable(self):
        
        self.tableExp = OWGUI.table(self.experimentTab, selectionMode=QTableWidget.MultiSelection)
        self.tableExp.setColumnCount(13)
        self.tableExp.setHorizontalHeaderLabels(["Name","Mean Mean","Stdev Mean","Min Mean", "Max Mean", "Mean Stdev", "Stdev Stdev", "Min Stdev", "Max Stdev", "Mean Cost", "Stdev Cost", "Min Cost", "Max Cost"])
        self.tableRun = OWGUI.table(self.runTab, selectionMode=QTableWidget.NoSelection)
        self.tableRun.setColumnCount(11)
        self.tableRun.setHorizontalHeaderLabels(["Name","Best","Cost"," evals", "argmin", "argmax", "min", "max", "mean", "stdev", "#exec"])

    def setTableRunDetail(self):
        print self.chosenRun

if __name__=="__main__":
    test_widget()

def test_widget():
    app = QApplication(sys.argv)
    w = GbOptTesterWidget()
    w.show()
    app.exec_()