from Goldenberry.classification.Perceptron import *
import Orange
from Goldenberry.classification.MulticlassLearner import OneVsAllMulticlassLearner, mask
import os
import itertools as itert

from unittest import *

class PerceptronTest(TestCase):
    """Perceptron tests."""

    def setUp(self):
        self.data_2d = Orange.data.Table(os.path.dirname(__file__) + "\\test_data_2d.tab")
        self.data_3d = Orange.data.Table(os.path.dirname(__file__) + "\\test_data_3d.tab")
        self.X, self.Y, _ = self.data_2d.to_numpy()
        self.Y = self.Y * 2 -1
    
    def test_basic(self):
        perceptron = Perceptron()
        while not perceptron.has_learned():
            perceptron.learn((self.X, self.Y))
        classes, scores = perceptron.predict(self.X)        
        self.assertTrue((self.Y == classes).all())
        self.assertTrue((scores != 0.0 ).all())

    def test_basic_neural(self):
        #data = Orange.data.Table("iris")
        data = self.data_2d
        learner = Orange.classification.neural.NeuralNetworkLearner()
        classifier = learner(data)
        for example in data:
            classifier(example)

    def test_perceptor_learner_base(self):
        max_iter = 5
        learner = PerceptronLearner(max_iter = max_iter)
        classifier = learner(self.data_2d)
        self.assertGreaterEqual(max_iter, learner.max_iter)
        self.assertIsNotNone(classifier.learner)
        self.assertIsNotNone(classifier.domain)

    def test_perceptor_learner_2d(self):
        max_iter = 10
        learner = PerceptronLearner(max_iter = max_iter)
        classifier = learner(self.data_2d)
        self.assertEquals(classifier.learner.__class__, Perceptron)

    def test_perceptor_learner_4d(self):
        max_iter = 10
        learner = PerceptronLearner(max_iter = max_iter)
        classifier = learner(self.data_3d)
        self.assertEquals(classifier.learner.__class__, OneVsAllMulticlassLearner)

    def test_perceptor_learner_classifier_2d(self):
        max_iter = 10
        learner = PerceptronLearner(max_iter = max_iter)
        classifier = learner(self.data_2d)
        for item in self.data_2d:
            value, prob = classifier(item, Orange.core.GetBoth)
            self.assertEqual(value, item.getclass())
            self.assertEquals(len(prob), 2)            

    def test_perceptor_learner_classifier_4d(self):
        max_iter = 20
        learner = PerceptronLearner(max_iter = max_iter)
        classifier = learner(self.data_3d)
        for item in self.data_3d:
            value, prob = classifier(item, Orange.core.GetBoth)
            self.assertEqual(value, item.getclass())
            self.assertEquals(len(prob), 4)

    def test_integration_perceptron_orange_learner(self):
        data = Orange.data.Table(os.path.dirname(__file__) + "\\test_data_2d.tab")
        max_iter = 10
        perceptron = PerceptronLearner(max_iter = max_iter)
        svm = Orange.classification.svm.SVMLearner()
        bayes = Orange.classification.bayes.NaiveLearner()
        learners = [svm, bayes, perceptron]
        results  = Orange.evaluation.testing.cross_validation(learners,data, folds = 10)
        
        print "Learner  CA     IS     Brier    AUC"
        for i in range(len(learners)):
            print "%-8s %5.3f  %5.3f  %5.3f  %5.3f" % (learners[i].name, \
            Orange.evaluation.scoring.CA(results)[i], Orange.evaluation.scoring.IS(results)[i],
            Orange.evaluation.scoring.Brier_score(results)[i], Orange.evaluation.scoring.AUC(results)[i])