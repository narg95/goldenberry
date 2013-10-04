"""
<name>Filter Attribute</name>
<description>Filters dataset's attributes.</description>
<contact>Nestor Rodriguez</contact>
<icon>icons/FilterAttributes.svg</icon>
<priority>1</priority>
"""

from Goldenberry.widgets import OWWidget, Orange
from Goldenberry.widgets import GbSolution, AttributeList, load_widget_ui, QStandardItem, QStandardItemModel, QString

class GbFilterAttributeWidget(OWWidget):
    """Provides a Filter Attributes widget."""

    def __init__(self, parent=None, signalManager=None, title = "Filter Attribute"):
        OWWidget.__init__(self, parent, signalManager, title)
        self.threshold = 1e-5
        self.setup_ui()
        self.setup_interfaces()

    def setup_ui(self):
        load_widget_ui(self)

    def setup_interfaces(self):
        self.inputs = [("Data", Orange.core.ExampleTable, self.set_data), ("Attributes Filter", GbSolution, self.set_solution)]

    def set_data(self, data):
        self.data = data
        if None is self.data:
            return

        model = QStandardItemModel()
        parent = model.invisibleRootItem()
        for att in self.data.domain.attributes:
            item = QStandardItem(QString(att.name))
            parent.appendRow(item)
            item.parent = parent

        self.attributesTree.setModel(model)

    def set_solution(self, solution):
        self.solution = solution

    def apply(self):
        if self.data is None or self.solution is None:
            return

        score = [ (att.name, self.solution.params[i])  for i, att in enumerate(self.data.domain.attributes)]
        newData = Orange.feature.selection.select_above_threshold(self.data, score, self.threshold)

        self.send("Data", newdata)
        self.send("Features", [ att.name for att in newData.domain.attributes])
