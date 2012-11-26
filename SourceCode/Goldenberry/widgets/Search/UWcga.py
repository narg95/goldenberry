from Orange.OrangeWidgets.OWWidget import OWWidget
from Orange.OrangeWidgets import OWGUI
from PyQt4.QtGui import QApplication
import sys

"""
<name>cGA</name>
<description>Compact Genetic Algorithm</description>
<contact>Nestor Rodriguez</contact>
<icon>icons/cGA.png</icon>
<priority>100</priority>
"""

class UWcga(OWWidget):
    """Widget for cga algorithm"""
    
    #attributes
    popsize = 20

    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'cGA')
        self.inputs = []
        self.outputs = []

        self.initgraphs()

    def initgraphs(self):
        box = OWGUI.widgetBox(self.controlArea, "Parameters")
        self.txtpopsize = OWGUI.lineEdit(box, self, "popsize", label="Population size", valueType = int)
        self.resize(100,50)

def TestWidget():
    app = QApplication(sys.argv)
    w = UWcga()
    w.show()
    app.exec_()