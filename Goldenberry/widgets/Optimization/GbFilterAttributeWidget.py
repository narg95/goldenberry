"""
<name>Feature subset and dependencies</name>
<description>Filters dataset's attributes.</description>
<contact>Nestor Rodriguez</contact>
<icon>icons/FilterAttributes.svg</icon>
<priority>1</priority>
"""

from Goldenberry.widgets import OWWidget, Orange, os, inspect, OWGUI, QHeaderView
from Goldenberry.widgets import GbSolution, AttributeList, load_widget_ui, QStandardItem, QStandardItemModel, QString, QIcon, QDoubleValidator, QIntValidator
import numpy as np

class GbFilterAttributeWidget(OWWidget):
    """Provides a Feature subset and dependencies widget."""

    settingsList = ['threshold', 'parttition_level']

    def __init__(self, parent=None, signalManager=None, title = "Feature subset and dependencies"):
        OWWidget.__init__(self, parent, signalManager, title)
        self.threshold = 1
        self.parttition_level = 1
        self.solution = None
        self.data = None
        self.att_items_list = None
        self.scores = None
        self.setup_ui()
        self.setup_interfaces()

    def setup_ui(self):
        load_widget_ui(self)
        path = os.path.dirname(inspect.getfile(type(self)))
        self.disabled_icon = QIcon(path + '/icons/disabled.png')
        self.enabled_icon = QIcon(path + '/icons/enabled.png')
        self.warning_label.setStyleSheet('color: red')

        parttionEditor = OWGUI.hSlider(self, self, 'parttition_level', minValue = 1, maxValue = 100, step = 1, label = "Partition Level")
        thresholdEditor = OWGUI.hSlider(self, self, 'threshold', minValue = 0, maxValue = 100, step = 1, divideFactor = 100.0, labelFormat = "%.3f", label = "Threshold")
        filter_button = OWGUI.button(self, self, "Filter",callback = self.filter_attributes)        
        
        self.actionswidgetlayout.addRow(parttionEditor.box, parttionEditor)
        self.actionswidgetlayout.addRow(thresholdEditor.box, thresholdEditor)        
        self.actionswidgetlayout.addWidget(filter_button)
        
        #tree views config
        self.attributesTree.header().setResizeMode(QHeaderView.ResizeToContents)
        self.filteredTree.header().setResizeMode(QHeaderView.ResizeToContents)

    def setup_interfaces(self):
        self.inputs = [("Data", Orange.core.ExampleTable, self.set_data), ("Attributes Filter", GbSolution, self.set_solution)]
        self.outputs = [("Data", Orange.core.ExampleTable), ("Features", AttributeList)]

    def set_data(self, data):
        self.data = data
        self.update_navigation()
        self.apply()

    def set_solution(self, solution):
        self.solution = solution
        self.update_navigation()
        self.apply()

    def update_navigation(self):
        if None is self.data or None is self.solution:
            return

        #check whether the amount of features is the same as scores.
        if len(self.data.domain.attributes) != len(self.solution.params):
            self.warning_label.setText("Number of features and scores does not match.")            
            return
        else:
            self.warning_label.setText("")
        
        self.update_attributes_navigation()
        self.update_filter_navigation()

    def update_attributes_navigation(self):
        model = QStandardItemModel()
        root_node = model.invisibleRootItem()
        
        # If there is a list of relevant attributes also known as the 'solution'
        if self.solution is not None and self.solution.params is not None:
            self.att_items_list = [QStandardItem(QString('%1 (%L2)').arg(att.name).arg(self.solution[idx])) \
                     for idx, att in enumerate(self.data.domain.attributes)]
            
            # Links roots with the invisible root node from the QStandardItemModel
            for root in self.solution.roots:
                root_node.appendRow(self.att_items_list[root])
                self.att_items_list[root].parent = root_node

            # Links children and parents
            for idx, children in enumerate(self.solution.children):
                for child_idx in children:
                    parent = self.att_items_list[idx]
                    child = self.att_items_list[child_idx]
                    parent.appendRow(child)
                    child.parent = parent
        else:
            for att in self.data.domain.attributes:
                item = QStandardItem(self.disabled_icon, att.name)
                root_node.appendRow(item)
                item.parent = root_node

        #updates the navigation view
        self.attributesTree.setModel(model)
        self.attributesTree.expandAll()

    def update_filter_navigation(self):
        self.filteredTree.setModel(None)
        self.scores = None
        
        if self.solution is None:
            return

        for item in self.att_items_list:
            item.setIcon(self.disabled_icon)

        model = QStandardItemModel()
        root_node = model.invisibleRootItem()
        self.scores = np.zeros(len(self.solution.params), dtype=int)
        for parent in self.solution.roots:
            for att_idx in self.list_attributes(parent, 1):
                if self.solution.params[att_idx] >= self.threshold/100.0:
                    item = QStandardItem(self.data.domain.attributes[att_idx].name)
                    root_node.appendRow(item)
                    item.parent = root_node
                    self.scores[att_idx] = 1.0
                    self.att_items_list[att_idx].setIcon(self.enabled_icon)

        #updates the navigation view
        self.filteredTree.setModel(model)
        self.filteredTree.expandAll()

    def list_attributes(self, parent, currlevel):
        if currlevel <= self.parttition_level:
            yield parent
            for child in self.solution.children[parent]:
                for att in self.list_attributes(child, currlevel + 1):
                    yield att

    def filter_attributes(self):
        self.update_navigation()
        self.apply()

    def apply(self):
        if self.data is None or self.solution is None or self.scores is None:
            return
        
        weighted_data = self.data.to_numpy("ac")[0] * np.concatenate((np.sqrt(self.solution.params), [1]))
        weighted_data = Orange.data.Table(self.data.domain, weighted_data)
        if self.scores.all():
            newdata = weighted_data            
        else:
            #filters the dataset
            score = [ (att.name, self.scores[i])  for i, att in enumerate(self.data.domain.attributes)]
            newdata = Orange.feature.selection.select_above_threshold(weighted_data, score, 1.0)
        
        #sends the signals with the data filtered
        self.send("Data", newdata)
        self.send("Features", [ att.name for att in newdata.domain.attributes])
