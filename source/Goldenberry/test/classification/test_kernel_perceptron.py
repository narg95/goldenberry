from Goldenberry.classification.KernelPerceptron import *
from Goldenberry.classification.Perceptron import *
from Goldenberry.classification.Kernels import *
from time import gmtime, strftime
import logging
import Orange
import os
import itertools as itert

from unittest import *

class KernelPerceptronTest(TestCase):
    """Kernel Perceptron Test"""

    def setUp(self):
        #self.training_set = Orange.data.Table(os.getcwd() + "\\data\\kclassesData.tab")
        #self.training_set = Orange.data.Table(os.getcwd() + "\\data\\testData.tab")
        #self.training_set = Orange.data.Table(os.getcwd() + "\\data\\notSeparable.tab")
        
        #X, z = load_data(os.getcwd() + "\\data\\realData10000.tab")
       
        logger = logging.getLogger('kernelPerceptron')
        hdlr = logging.FileHandler('kernelPerceptron.log')
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr) 
        logger.setLevel(logging.DEBUG)
        
        self.logger = logger
        

        self.data = Orange.data.Table(os.getcwd() + "\\data\\realData10000.tab")
        
        #folds = 4
        #cv_indices = Orange.data.sample.SubsetIndicesCV(data, folds)

        #for fold in range(1):#range(folds):
        #    train = data.select(cv_indices, fold, negate = 1)
        #    test  = data.select(cv_indices, fold)
        
        #self.training_set = train
        #self.testing_set = test
        #self.X, self.Y, _ = self.training_set.to_numpy()

        #self.testing_set = self.training_set.random_example()
        #self.validation_set = None
        #self.X, self.Y, _ = self.training_set.to_numpy()
        #RX, rz, VX, vz, TX, tz = train_test_val_split(self.X, self.Y, 0.5, 0.25, 0.25)


    def _test_kernelPerceptron_two_classes(self):
        self.training_set = Orange.data.Table(os.getcwd() + "\\data\\testData.tab")
        self.X, self.Y, _ = self.training_set.to_numpy()
        kernel = LinealKernel()
        kernelperceptron = KernelPerceptron(kernel, None)
        K = None
        sv_x, sv_y, sv_alpha = None, None, None
        
        #adds a column of ones
        Xdata = np.hstack((np.ones((self.X.shape[0], 1)), self.X))
        Yclass = np.zeros(self.Y.shape)   
        # Set the Y class to -1, instead of 0.     
        for i in range(len(Yclass)):
            Yclass[i] = -1 if self.Y[i] == 0 else self.Y[i]

        iterations = 0
        while None == K or K > 0 and iterations < 100:
            sv_x, sv_y, sv_alpha, K = kernelperceptron.learn((Xdata, Yclass),(sv_x, sv_y, sv_alpha))
            iterations += 1
        
        prediction = kernelperceptron.predict(Xdata, (sv_x, sv_y, sv_alpha))
        for yp, yi in zip(prediction, Yclass):
            self.assertEqual(yp[0], yi)

        print("\n KernelPerceptron ends after {0} iterations ".format(iterations))

    def _test_perceptor_learner(self):
        max_iter = 1
        learner = KernelPerceptronLearner(max_iter = max_iter)
        classifier = learner(self.training_set)
        self.assertGreaterEqual(max_iter, learner.iters)
        self.assertIsNotNone(classifier.predict)
        self.assertIsNotNone(classifier.domain)

    def test_perceptor_classifier(self):
        max_iter = 1000
        folds = 4
        cv_indices = Orange.data.sample.SubsetIndicesCV(self.data, folds)
        folds = 1
        for fold in range(folds):
            print "Fold %d: train " % fold
            self.logger.info("Fold %d: train " % fold)
            train = self.data.select(cv_indices, fold, negate = 1)
            test  = self.data.select(cv_indices, fold)
        
            self.logger.info("Creating Learner... " )
            learner = KernelPerceptronLearner(max_iter = max_iter)
            self.logger.info("Learning... " )
            classifier = learner(train)
            errors = 0
            success = 0
            self.logger.info("Classifying... ")
            for item in test: #self.training_set:
                result = classifier(item)
                if result == item.getclass():
                    success += 1
                else:
                    errors += 1
            
            self.logger.info("End classifying... " )
            msg = "KernelPerceptron ends with {0} errors and {1} successes using {2} learning iterations".format(errors, success, max_iter)
            self.logger.info(msg)
            print(msg)
            print
            #self.assertGreaterEqual(success*0.20, errors)

    def _test_value_by_ref(self):
        array = np.array([1,2,3,4,5,6])
        print array
        self.changeValues(array)
        print array

    def changeValues(self, array):
        array[1] = 999
        array = np.hstack( (0, array))

    def load_data(fn):
        with open(fn) as fp:
            lines = fp.readlines()
        # Remove whitespace.
        lines = [i.strip() for i in lines]
        # Remove empty lines.
        lines = [i for i in lines if i]
        # Split by comma.
        lines = [i.split(',') for i in lines]
        # Inputs are the first four elements.
        inpts = [i[:4] for i in lines]
        # Labels are the last.
        labels = [i[-1] for i in lines]

        # Make arrays out of the inputs, one row per sample.
        X = np.empty((150, 4))
        X[:] = inpts

        # Make integers array out of label strings.
        #
        # We do this by first creating a set out of all labels to remove
        # any duplicates. Then we create a dictionary which maps label
        # names to an index. Afterwards, we loop over all labels and
        # assign the corresponding integer to that field in the label array z.
        z = np.empty(150)
        label_names = sorted(set(labels))
        label_to_idx = dict((j, i) for i, j in enumerate(label_names))

        for i, label in enumerate(labels):
            z[i] = label_to_idx[label]

        return X, z

    def train_test_val_split(X, Z, train_frac, val_frac, test_frac):
        """Split the data into three sub data sets, one for training, one for
        validation and one for testing. The data is shuffled first."""
        assert train_frac + val_frac + test_frac == 1, "fractions don't sum up to 1"

        n_samples = X.shape[0]
        n_samples_train = int(math.floor(n_samples * train_frac))
        n_samples_val = int(math.floor(n_samples * val_frac))

        idxs = range(n_samples)
        random.shuffle(idxs)
        train_idxs = idxs[:n_samples_train]
        val_idxs = idxs[n_samples_train:n_samples_train + n_samples_val]
        test_idxs = idxs[n_samples_train + n_samples_val:]

        return (X[train_idxs], Z[train_idxs],
                X[val_idxs], Z[val_idxs],
                X[test_idxs], Z[test_idxs])