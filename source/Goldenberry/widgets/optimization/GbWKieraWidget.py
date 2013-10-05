"""
<name>WKiera Cost Function</name>
<description>This is a cost function to facilitate the creation of a WKiera algorithm.</description>
<contact>Nestor Rodriguez</contact>
<icon>icons/Pbil.svg</icon>
<priority>200</priority>

"""

from Goldenberry.widgets import OWWidget, GbCostFunction, GbKernel, WKieraCostFunction, GbKernel
from Orange.core import Learner
from orange import ExampleTable
class GbWKieraWidget(OWWidget):
    """WKiera cost function widget."""
    data = None
    kernel = None
    learner = None

    def __init__(self, parent=None, signalManager=None, title="WKiera Cost Function"):
        OWWidget.__init__(self, parent, signalManager, title)
        self.setup_interfaces()

    def setup_interfaces(self):
        self.outputs = [("Cost Function", GbCostFunction)]
        self.inputs = [("Kernel Function", GbKernel, self.set_kernel),
                       ("Learner", Learner, self.set_learner ),
                       ("Data", ExampleTable, self.set_data)]

    def set_kernel(self, kernel):
        self.kernel = kernel
        self.apply()

    def set_learner(self, learner):
        self.learner = learner
        self.apply()

    def set_data(self, data):
        self.data = data
        self.apply()

    def apply(self):
        if self.kernel is not None and self.learner is not None and self.data is not None:
            wkiera_cost_func = WKieraCostFunction(GbKernel(*self.kernel), self.data, self.learner)
            self.send("Cost Function", wkiera_cost_func)
            
    