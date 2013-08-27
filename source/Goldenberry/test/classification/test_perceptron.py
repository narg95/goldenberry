from Goldenberry.classification.Perceptron import *
import Orange
import os
import itertools as itert

from unittest import *

class PerceptronTest(TestCase):
    """Perceptron tests."""

    def setUp(self):
        self.training_set = Orange.data.Table(os.path.dirname(__file__) + "\\test_data_2d.tab")
        self.X, self.Y, _ = self.training_set.to_numpy()
    
    def test_basic(self):
        perceptron = Perceptron()
        W, B, K = None, 0, None
        while None == K or K > 0:
            W, B, K = perceptron.learn((self.X, self.Y),(W, B))
        prediction = perceptron.predict(self.X, (W, B))

        for yp, yi in itert.imap(lambda y1, y2 : (y1, -1 if y2 == 0 else y2), prediction, self.Y):
            self.assertEqual(yp, yi)

    def test_perceptor_learner(self):
        max_iter = 5
        learner = PerceptronLearner(max_iter = max_iter)
        classifier = learner(self.training_set)
        self.assertGreaterEqual(max_iter, learner.iters)
        self.assertIsNotNone(classifier.W)
        self.assertIsNotNone(classifier.B)
        self.assertIsNotNone(classifier.predict)
        self.assertIsNotNone(classifier.domain)

    def test_perceptor_classifier(self):
        max_iter = 5
        learner = PerceptronLearner(max_iter = max_iter)
        classifier = learner(self.training_set)
        for item in self.training_set:
            self.assertEqual(classifier(item), item.getclass())

    def test_integration_perceptron_orange_learner(self):
        data = Orange.data.Table(os.path.dirname(__file__) + "\\test_data_2d.tab")
        max_iter = 5
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

    def test_multiclass_one_vs_one_learner():
        data = Orange.data.Table(os.path.dirname(__file__) + "\\test_data_2d.tab")
