import Orange
from unittest import *
from Goldenberry.widgets import QtCore, Kernels
from Goldenberry.classification.base.GbKernel import GbKernel
from Goldenberry.classification.base.GbFactory import GbFactory
from Goldenberry.optimization.cost_functions import *
from Goldenberry.optimization.edas.Univariate import *
from Goldenberry.optimization.base.GbCostFunction import *
from PyQt4.QtGui import QApplication
from Goldenberry.classification.Perceptron import PerceptronLearner
from Goldenberry.widgets.optimization.GbCgaWidget import GbCgaWidget
from Goldenberry.widgets.optimization.GbBmdaWidget import GbBmdaWidget
from Goldenberry.widgets.optimization.GbUmdaWidget import GbUmdaWidget
from Goldenberry.widgets.optimization.GbPbilWidget import GbPbilWidget
from Goldenberry.widgets.optimization.GbCostFuncsWidget import GbCostFuncsWidget
from Goldenberry.widgets.optimization.GbTildaWidget import GbTildaWidget
from Goldenberry.widgets.optimization.GbBlackBoxWidget import GbBlackBoxWidget
from Goldenberry.widgets.learners.GbKernelBuilderWidget import GbKernelBuilderWidget
from Goldenberry.widgets.learners.GbPerceptronWidget import GbPerceptronWidget
from Goldenberry.widgets.learners.GbSvmWidget import GbSvmWidget
from Goldenberry.widgets.optimization.GbWKieraWidget import GbWKieraWidget
from Goldenberry.widgets.optimization.GbFilterAttributeWidget import GbFilterAttributeWidget

import sys

class WidgetsTest(TestCase):
    """Test the widgets for the optimization part."""

    def setUp(self):
        self.app = QApplication(sys.argv)

    def test_cga_basic(self):        
        pass