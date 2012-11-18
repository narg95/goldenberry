from Orange.OrangeWidgets.OWWidget import OWWidget
from Orange.OrangeWidgets import OWGUI
from PyQt4.QtGui import QApplication
from MLFw.fitfunctions import fitness
import sys
import inspect
import types

"""
<name>Fitness</name>
<description>Provide a set of fitness functions</description>
<contact>Nestor Rodriguez</contact>
<icon>icons/cGA.png</icon>
<priority>200</priority>
"""

class OWFitFuncs(OWWidget):
    """Widget for fitness functions"""
    
    fitfuncs = dict(inspect.getmembers(fitness, inspect.isfunction))
    fitselected = None
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'fitness')
        self.outputs = [("Fitness Functions", baseFitness)]
        self.initgraphs()

    def initgraphs(self):
        box = OWGUI.widgetBox(self.controlArea, "Function Selection") 
        OWGUI.comboBox(box, self, "fitselected", label="Functions: ",sendSelectedValue = 1, items=self.fitfuncs.keys(), control2attributeDict = self.fitfuncs)
        self.okbutt = OWGUI.button(box, self, "Ok", callback = self.oksend)
    
    def oksend(self):
        self.send("Fitness", self.fitselected)

def TestWidget():
    app = QApplication(sys.argv)
    w = OWFitnessFunctions()
    w.show()
    app.exec_()