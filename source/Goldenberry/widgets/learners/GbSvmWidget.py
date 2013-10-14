# coding=utf-8
"""
<name>SVM</name>
<description>Support Vector Machines learner/classifier.</description>
<icon>icons/SVM.svg</icon>
<priority>100</priority>
"""

from Orange.OrangeWidgets.Classify.OWSVM import OWSVM, UnhandledException
from Goldenberry.widgets import GbKernel, QCheckBox, QString, GbFactory
from Orange.OrangeWidgets.OWGUI import appendRadioButton
from orange import ExampleTable, Learner, Classifier
from orngWrap import PreprocessedLearner
from Orange.OrangeWidgets.OWWidget import Default
import orange, orngSVM
import numpy as np

class GbSvmWidget(OWSVM):
    
    kernel_func = None
    kernel_type = orngSVM.SVMLearner.Linear

    def __init__(self, parent=None, signalManager=None):
        super(GbSvmWidget, self).__init__(parent=parent, signalManager=signalManager, name="SVM")
        appendRadioButton(self.kernelradio, self,"kernel_type","Custom")
        self.inputs = [('Kernel Function', GbKernel, self.set_kernel), ("Data", ExampleTable, self.setData)]

        self.outputs = [("Learner", Learner, Default),
                        ("Learner Factory", GbFactory, Default),
                        ("Classifier", Classifier, Default),
                        ("Support Vectors", ExampleTable)]

        self.settingsList.append("kernel_func") 
        self.kernel_type = orngSVM.SVMLearner.Custom
        self.normalization=0
        for check in self.findChildren(QCheckBox):
            if check.text() == QString("Normalize data"):
                check.close()

    def set_kernel(self, kernel):
        kernel = kernel(None)
        self.kernel_func = lambda x,y: kernel(*to_numpy(x, y))

    def applySettings(self):
        if self.kernel_type == orngSVM.SVMLearner.Custom and None is self.kernel_func:
            return

        self.learner=orngSVM.SVMLearner()
        parameters = {}
        for attr in ("name", "kernel_type", "kernel_func", "degree", "shrinking", "probability", "normalization"):
            setattr(self.learner, attr, getattr(self, attr))
            parameters[attr] = getattr(self, attr)

        for attr in ("gamma", "coef0", "C", "p", "eps", "nu"):
            setattr(self.learner, attr, float(getattr(self, attr)))
            parameters[attr] = getattr(self, attr)

        self.learner.svm_type=orngSVM.SVMLearner.C_SVC

        if self.useNu:
            self.learner.svm_type=orngSVM.SVMLearner.Nu_SVC
            parameters["svm_type"] = orngSVM.SVMLearner.Nu_SVC
        
        self.classifier=None
        self.supportVectors=None
        
        if self.data:
            if self.data.domain.classVar.varType==orange.VarTypes.Continuous:
                self.learner.svm_type+=3
            self.classifier=self.learner(self.data)
            self.supportVectors=self.classifier.supportVectors
            self.classifier.name=self.name
            
        self.send("Learner", self.learner)
        self.send("Learner Factory", GbFactory(orngSVM.SVMLearner, parameters))
        self.send("Classifier", self.classifier)
        self.send("Support Vectors", self.supportVectors) 

def to_numpy(i1, i2):
    x = np.array([i.value for i in i1 if i.value != i1.get_class().value], dtype = float)
    y = np.array([i.value for i in i2 if i.value != i2.get_class().value], dtype = float)
    return x,y