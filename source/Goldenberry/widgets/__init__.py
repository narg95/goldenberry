import Orange
from orngWrap import PreprocessedLearner
from Orange.OrangeWidgets.OWWidget import OWWidget, Multiple
from Orange.OrangeWidgets import OWGUI
from PyQt4.QtGui import QApplication, QIntValidator, QLabel, QWidget, QFormLayout, QAbstractButton, QDoubleValidator, QTableWidget, QClipboard
from PyQt4.QtCore import QObject, Qt
from PyQt4 import QtCore, QtGui, uic
import os
import sys
import abc
import inspect
import types
# optimization imports
from Goldenberry.optimization.base.GbCostFunction import *
from Goldenberry.base.GbDynamicFunction import *
from Goldenberry.classification.base.GbKernel import GbKernel
import Goldenberry.classification.Kernels as Kernels
from Goldenberry.optimization.base.GbBaseOptimizer import *
from Goldenberry.optimization.edas.GbBlackBoxTester import GbBlackBoxTester
import Goldenberry.optimization.cost_functions as cost_functions
from Goldenberry.optimization.edas.Univariate import Cga, Pbil, Tilda, Pbilc
from Goldenberry.optimization.edas.Bivariate import Bmda, DependencyMethod
from Goldenberry.feature_selection.WKiera import *

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