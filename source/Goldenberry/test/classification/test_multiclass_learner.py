from Goldenberry.classification.Perceptron import *
from Goldenberry.classification.MulticlassLearner import *

import Orange
import os
import itertools as itert

from unittest import *

class MulticlassLearnerTest(TestCase):
    """Multiclass learners tests."""

    def setUp(self):
        self.training_set = Orange.data.Table(os.path.dirname(__file__) + "\\test_data_3d.tab")
        self.X, self.Y, _ = self.training_set.to_numpy()
        self.n_classes = len(self.training_set.domain.class_var.values)
    
    def test_multiclass_one_vs_one_learner_basic(self):
        m_learner = OneVsAllMulticlassLearner(Perceptron, self.n_classes)
        self.assertEquals(len(m_learner.learners), self.n_classes)
        self.assertEquals(m_learner.nY.shape ,(self.n_classes, self.n_classes))
        self.assertEquals((m_learner.nY > 0).sum(),self.n_classes)
        self.assertEquals((m_learner.nY < 0).sum(), (self.n_classes * self.n_classes - self.n_classes))
        
    def test_multiclass_one_vs_one_mask(self):
        m_learner = OneVsAllMulticlassLearner(Perceptron, 4)
        Y = np.array([0,1,2,3])
        self.assertTrue((mask(m_learner.nY, Y, 0) == [1,-1,-1,-1]).all())
        self.assertTrue((mask(m_learner.nY, Y, 1) == [-1,1,-1,-1]).all())
        self.assertTrue((mask(m_learner.nY, Y, 2) == [-1,-1,1,-1]).all())
        self.assertTrue((mask(m_learner.nY, Y, 3) == [-1,-1,-1,1]).all()) 

    def test_multiclass_one_vs_one_learn(self):
        m_learner = OneVsAllMulticlassLearner(Perceptron, self.n_classes)
        m_learner.learn((self.X, self.Y))
        for learner in m_learner.learners:
            self.assertTrue(learner.iters > 0)
            self.assertTrue(learner.iters == m_learner.iters)

    def test_multiclass_one_vs_one_predict(self):
        m_learner = OneVsAllMulticlassLearner(Perceptron, self.n_classes)
        m_learner.learn((self.X, self.Y))
        while not m_learner.has_learned():
            m_learner.learn((self.X, self.Y))
        
        predictions, scores = m_learner.predict(self.X)
        self.assertTrue((predictions == self.Y).all())