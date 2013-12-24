import sys
sys.path.append('..')

import numpy as np
import scipy
import random
import urllib

from common.model import Model
from common.question import Question

class LinearRegression(Model):
    def __init__(self, training_set=None, testing_set=None):
        Model.__init__(self, training_set, testing_set)

    def train(self, labels, k=None):
        data_matrix = np.array(self.training_set)
        target_vector = labels

        # normal equations linear regression
        pinv = np.linalg.pinv(data_matrix)
        self.weights = np.dot(pinv, target_vector)

        if k is not None:
            lam = 10 ** k
            term = np.dot(self.weights.T[0], self.weights.T[0]) * (lam / len(data_matrix))
            regularized = np.linalg.pinv(data_matrix + term)
            self.weights = np.dot(regularized, target_vector)

    def test(self, labels):
        targets = labels

        tys = np.sign(np.dot(self.testing_set, self.weights))
        return len(targets[tys != targets]) / float(len(targets))

def get_data(url):
    f = urllib.urlopen(url)
    data = [[float(value) for value in line.strip('\n').split('\r')[0].split()] for line in f]
    training = [e[:-1] for e in data]
    labels = [[e[-1]] for e in data]
    return np.array(training), np.array(labels)

def test_run(test_runs, k=None):
    avg_in, avg_out = 0, 0

    training_set, training_labels = get_data("http://work.caltech.edu/data/in.dta")
    testing_set, testing_labels = get_data("http://work.caltech.edu/data/out.dta")

    transform = lambda x: [1, x[0], x[1], x[0] ** 2, x[1] ** 2, x[0] * x[1], np.abs(x[0] - x[1]), np.abs([x[0] + x[1]])]

    transformed_training_set = [transform(feature) for feature in training_set]
    transformed_testing_set = [transform(feature) for feature in testing_set]

    for i in xrange(test_runs):
        lr = LinearRegression(transformed_training_set, transformed_training_set)

        if k is not None:
            lr.train(training_labels, k)
        else:
            lr.train(training_labels)

        avg_in += lr.test(training_labels)

        lr.testing_set = transformed_testing_set
        avg_out += lr.test(testing_labels)

    avg_in /= float(test_runs)
    avg_out /= float(test_runs)
    return (avg_in, avg_out)

if __name__ == "__main__":
    question2 = Question("2. no regularization (in-sample, out-of-sample)",
                         [(0.03, 0.08),
                          (0.03, 0.10),
                          (0.04, 0.09),
                          (0.04, 0.11),
                          (0.05, 0.10)], 'a',
                         lambda result, choice: abs(result[0] - choice[0]) + abs(result[1] - choice[1]))

    question2.check(test_run(100))

    question3 = Question("3. weight decay (in-sample, out-of-sample)",
                         [(0.01, 0.02),
                          (0.02, 0.04),
                          (0.02, 0.06),
                          (0.03, 0.08),
                          (0.03, 0.10)], 'd',
                         lambda result, choice: abs(result[0] - choice[0]) + abs(result[1] - choice[1]))

    question3.check(test_run(100, -3))

    question4 = Question("4. k = 3 (in-sample, out-of-sample)",
                         [(0.2, 0.2),
                          (0.2, 0.3),
                          (0.3, 0.3),
                          (0.3, 0.4),
                          (0.4, 0.4)], 'e',
                         lambda result, choice: abs(result[0] - choice[0]) + abs(result[1] - choice[1]))

    question4.check(test_run(100, 3))

    question5 = Question("5. smallest in-sample k",
                         [2, 1, 0, -1, -2], 'd')

    choices = [2, 1, 0, -1, -2]
    errors = [test_run(100, k) for k in choices]
    zipped = zip(choices, errors)
    answer = min(zipped, key=lambda c: c[1][1])[0]
    question5.check(answer)

    question6 = Question("6. smallest out-of-sample k",
                         [0.04, 0.06, 0.08, 0.10, 0.12], 'b')

    choices = range(-20, 20 + 1)
    errors = [test_run(100, k) for k in choices]
    zipped = zip(choices, errors)
    answer = min(zipped, key=lambda c: c[1][1])[1][1]
    question6.check(answer)
