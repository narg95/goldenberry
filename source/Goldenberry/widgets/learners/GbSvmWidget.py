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
from Goldenberry.classification.SvmLearner import SvmLearner
import numpy as np

class GbSvmWidget(OWSVM):
    
    kernel = None
    kernel_type = orngSVM.SVMLearner.Custom

    def __init__(self, parent=None, signalManager=None):
        super(GbSvmWidget, self).__init__(parent=parent, signalManager=signalManager, name="SVM")
        appendRadioButton(self.kernelradio, self,"kernel_type","Custom")
        self.inputs = [('Kernel Function', GbKernel, self.set_kernel), ("Data", ExampleTable, self.setData),
                       ("Preprocess", PreprocessedLearner, self.setPreprocessor)]

        self.outputs = [("Learner", Learner, Default),
                        ("Classifier", Classifier, Default),
                        ("Support Vectors", ExampleTable)]

        self.settingsList.append("kernel") 
        self.controlArea.layout().removeWidget(self.kernelradio)
        self.kernelradio.close()
        self.kernel_type = orngSVM.SVMLearner.Custom

    def set_kernel(self, kernel):
        self.kernel = kernel(None)

    def applySettings(self):        

        self.learner= SvmLearner()
        self.kernel_type = orngSVM.SVMLearner.Custom

        for attr in ("name", "kernel_type", "kernel", "degree", "shrinking", "probability", "normalization"):
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
        
        if self.data and self.kernel:
            if self.data.domain.classVar.varType==orange.VarTypes.Continuous:
                self.learner.svm_type+=3
            
            self.classifier=self.learner(self.data)
            self.supportVectors=self.classifier.supportVectors
            self.classifier.name=self.name
            
        self.send("Learner", self.learner)
        self.send("Classifier", self.classifier)
        self.send("Support Vectors", self.supportVectors)
    
    def search_(self):
        learner=SvmLearner()
        for attr in ("name", "kernel_type", "degree", "kernel" ,"shrinking", "probability", "normalization"):
            setattr(learner, attr, getattr(self, attr))

        for attr in ("gamma", "coef0", "C", "p", "eps", "nu"):
            setattr(learner, attr, float(getattr(self, attr)))

        learner.svm_type=0

        if self.useNu:
            learner.svm_type=1
        params=[]
        if self.useNu:
            params.append("nu")
        else:
            params.append("C")
        try:
            learner.tuneParameters(self.data, params, 4, verbose=0,
                                   progressCallback=self.progres)
        except UnhandledException:
            pass
        for param in params:
            setattr(self, param, getattr(learner, param))
            
        self.finishSearch()  
