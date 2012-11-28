from Goldenberry.classification.Perceptron import Perceptron
import Orange
import os
import itertools as itert

from unittest import *

class PerceptronTest(TestCase):
    """Perceptron tests."""

    def setUp(self):
        self.X, self.Y, _ = Orange.data.Table(os.path.dirname(__file__) + "\\test_date_2d.tab").to_numpy()
    
    def test_basic(self):
        perceptron = Perceptron()
        W, B, K = None, 0, None
        while None == K or K > 0:
            W, B, K = perceptron.learn((self.X, self.Y),(W, B))
        prediction = perceptron.predict(self.X, (W, B))

        for yp, yi in itert.imap(lambda y1, y2 : (y1, -1 if y2 == 0 else y2), prediction, self.Y):
            self.assertEqual(yp, yi)

