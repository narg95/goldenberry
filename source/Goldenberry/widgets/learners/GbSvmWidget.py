# coding=utf-8
"""
<name>SVM</name>
<description>Support Vector Machines learner/classifier.</description>
<icon>icons/SVM.svg</icon>
<priority>100</priority>
"""

from Orange.OrangeWidgets.Classify.OWSVM import OWSVM, UnhandledException
from Goldenberry.widgets import GbKernel
from Orange.OrangeWidgets.OWGUI import appendRadioButton
from orange import ExampleTable, Learner, Classifier
from orngWrap import PreprocessedLearner
from Orange.OrangeWidgets.OWWidget import Default
import orange, orngSVM
import numpy as np

class GbSvmWidget(OWSVM):
    
    kernel_func = None

    def __init__(self, parent=None, signalManager=None):
        super(GbSvmWidget, self).__init__(parent=parent, signalManager=None, name="SVM")
        appendRadioButton(self.kernelradio, self,"kernel_type","Custom")
        self.inputs = [('Kernel Function', GbKernel, self.set_kernel), ("Data", ExampleTable, self.setData),
                       ("Preprocess", PreprocessedLearner, self.setPreprocessor)]

        self.outputs = [("Learner", Learner, Default),
                        ("Classifier", Classifier, Default),
                        ("Support Vectors", ExampleTable)]

        self.settingsList.append("kernel_func")        

    def set_kernel(self, kernel):
        self.kernel_func = orngSVM.KernelWrapper(lambda x,y: kernel(None).execute(*to_numpy(x,y)))

    def applySettings(self):
        self.learner=orngSVM.SVMLearner()
        for attr in ("name", "kernel_type", "kernel_func", "degree", "shrinking", "probability", "normalization"):
            setattr(self.learner, attr, getattr(self, attr))

        for attr in ("gamma", "coef0", "C", "p", "eps", "nu"):
            setattr(self.learner, attr, float(getattr(self, attr)))

        self.learner.svm_type=orngSVM.SVMLearner.C_SVC

        if self.useNu:
            self.learner.svm_type=orngSVM.SVMLearner.Nu_SVC

        if self.preprocessor:
            self.learner = self.preprocessor.wrapLearner(self.learner)
        self.classifier=None
        self.supportVectors=None
        
        if self.data:
            if self.data.domain.classVar.varType==orange.VarTypes.Continuous:
                self.learner.svm_type+=3

            self.learner.kernel_func = self.kernel_func
            self.classifier=self.learner(self.data)
            self.supportVectors=self.classifier.supportVectors
            self.classifier.name=self.name
            
        self.send("Learner", self.learner)
        self.send("Classifier", self.classifier)
        self.send("Support Vectors", self.supportVectors)
      
    def search_(self):
        learner=orngSVM.SVMLearner()
        learner.kernel_func = self.kernel_func
        learner.kernel_type = 4
        learner.progressCallback = self.progres

        self.finishSearch()

def to_numpy(i1, i2):
    x = np.array([i for i in i1 if i.native() != i1.get_class().native()], dtype = float)
    y = np.array([i for i in i2 if i.native() != i2.get_class().native()], dtype = float)
    return x,y