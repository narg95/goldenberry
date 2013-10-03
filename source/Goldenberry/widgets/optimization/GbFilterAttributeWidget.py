"""
<name>Filter Attribute</name>
<description>Filters dataset's attributes.</description>
<contact>Nestor Rodriguez</contact>
<icon>icons/FilterAttributes.svg</icon>
<priority>1</priority>
"""

from Goldenberry.widgets import OWWidget, Orange
from Goldenberry.widgets import GbSolution

class GbFilterAttributeWidget(OWWidget):
    """Provides a Filter Attributes widget."""
   
    def __init__(self, parent=None, signalManager=None):
        super(GbFilterAttributeWidget, self).__init__(parent, signalManager, 'Filter Attribute', "Filter Attribute")

    def setup_interfaces(self):
        self.inputs = [("Data", Orange.core.ExampleTable, self.set_data), ("Attributes Filter", GbSolution, self.set_filter)]
        self.outputs = [("Data", Orange.core.ExampleTable)]

    def set_data(self, data):
        self.data = data

    def set_filter(self, filter):
        filter.params
