import Orange
from orngWrap import PreprocessedLearner
from Orange.OrangeWidgets.OWWidget import OWWidget
from Orange.OrangeWidgets import OWGUI
from PyQt4.QtGui import QApplication, QIntValidator, QLabel, QWidget, QFormLayout, QAbstractButton, QDoubleValidator
from PyQt4.QtCore import QObject
from PyQt4 import QtCore
from PyQt4 import uic
import os
import sys
import abc
import inspect
import types

# optimization imports
from Goldenberry.optimization.base.GbBaseCostFunction import *
from Goldenberry.optimization.base.GbBaseOptimizer import *
from Goldenberry.optimization.cost_functions.functions import *
from Goldenberry.optimization.cost_functions import functions
from Goldenberry.optimization.edas.Univariated import Cga, Pbil
from Goldenberry.optimization.edas.Bivariated import Bmda

# classification imports
from Goldenberry.classification.Perceptron import PerceptronLearner, PerceptronClassifier

def load_widget_ui(widget):
    """Loads the widget's QT Ui file (widget.ui). The file must be located in the same
    directory of the widget and with the same name with the .ui extension."""
    widget_type = type(widget)
    path = os.path.dirname(inspect.getfile(widget_type)) + "\\" + widget_type.__name__ + ".ui"
    if os.path.isfile(path):
        widget.controlArea = uic.loadUi(path, widget)
    else:
        raise Exception("Ui file not found for widget: " + str(widget))