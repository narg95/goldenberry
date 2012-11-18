from Orange.OrangeWidgets.OWWidget import OWWidget
from Orange.OrangeWidgets import OWGUI
from PyQt4.QtGui import QApplication, QIntValidator, QLabel, QWidget, QFormLayout, QAbstractButton
from PyQt4.QtCore import QObject
from PyQt4 import QtCore
from PyQt4 import uic
from MLFw.searchers.baseSearcher import *
from MLFw.fitfunctions.baseFitness import *
from MLFw.fitfunctions.fitnessFuncs import *
from MLFw.misc import *
import os
import sys

#Interface Name Definitions
class _interfaces(object):
    @constant
    def FITNESS_FUNCTION():
        return "Fitness Function", baseFitness
    @constant
    def SEARCH_ALGORITHM():
        return "Search Algorithm", baseSearcher
 
INTERFACES = _interfaces()
