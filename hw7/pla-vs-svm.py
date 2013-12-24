import sys
sys.path.append('..')

import numpy as np
import scipy
import random
import urllib

from sklearn import svm
from sklearn.grid_search import GridSearchCV

from common.model import Model
from common.question import Question

from hw1.perceptron import Perceptron

def test_run(runs, size):
    support_vectors = []
    svm_wins = 0
    run = 0

    while run < runs:
        print run
        training_set = np.array([[1., random.uniform(-1, 1), random.uniform(-1, 1)]
                        for i in xrange(size)])
        testing_set = np.array([[1., random.uniform(-1, 1), random.uniform(-1, 1)]
                       for i in xrange(1000)])

        pla = Perceptron(training_set, testing_set)

        targets = np.array([pla.target(feature) for feature in training_set])

        # if they're all on the same side, skip
        if np.all(targets == -1.) or np.all(targets == 1.):
            continue

        # perceptron
        iterations = pla.train()
        perceptron_error = pla.test()

        # svm
        machine = svm.SVC(kernel='linear', C=1.0e6)
        svm_data = np.delete(training_set, 0, 1)
        machine.fit(svm_data, targets)
        svm_error = 0.0

        svm_predict = np.delete(testing_set, 0, 1)
        predictions = machine.predict(svm_predict)
        targets = np.array([pla.target(feature) for feature in testing_set])
        svm_error = len(targets[predictions != targets]) / float(len(targets))

        if svm_error < perceptron_error:
            svm_wins += 1
            support_vectors += [np.sum(machine.n_support_)]

        run += 1

    return (svm_wins / float(runs), np.mean(support_vectors))

if __name__ == "__main__":
    question8 = Question("8. [n = 10] how often svm better than pla",
                         [.2, .4, .6, .8, 1.], 'c')

    answer = test_run(1000, 10)
    question8.check(answer[0])

    question9 = Question("9. [n = 100] how often svm better than pla",
                         [.1, .3, .5, .7, .9], 'd')

    answer = test_run(1000, 100)
    question9.check(answer[0])

    question10 = Question("10. [n = 100] average number of support vectors of svm",
                         [2, 3, 5, 10, 20], 'b')

    question10.check(answer[1])

