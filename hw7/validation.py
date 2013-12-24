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

    def train(self, labels):
        data_matrix = np.array(self.training_set)
        target_vector = labels

        # normal equations linear regression
        pinv = np.linalg.pinv(data_matrix)
        self.weights = np.dot(pinv, target_vector)

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

def test_run(k, reverse=False):
    avg_in, avg_out, avg_classification = 0.0, 0.0, 0.0

    in_data = get_data("http://work.caltech.edu/data/in.dta")
    training_set, training_labels = in_data[0][:25], in_data[1][:25]
    validation_set, validation_labels = in_data[0][25:], in_data[1][25:]

    if reverse:
        training_set, validation_set = validation_set, training_set
        training_labels, validation_labels = validation_labels, training_labels

    testing_set, testing_labels = get_data("http://work.caltech.edu/data/out.dta")

    transform = lambda x: [1, x[0], x[1], x[0] ** 2, x[1] ** 2, x[0] * x[1], np.abs(x[0] - x[1]), np.abs([x[0] + x[1]])]

    transformed_training_set = [transform(feature) for feature in training_set]
    transformed_validation_set = [transform(feature) for feature in validation_set]
    transformed_testing_set = [transform(feature) for feature in testing_set]

    model = lambda x: x[:k + 1]

    cut_training_set = [model(feature) for feature in transformed_training_set]
    cut_validation_set = [model(feature) for feature in transformed_validation_set]
    cut_testing_set = [model(feature) for feature in transformed_testing_set]

    lr = LinearRegression(cut_training_set, cut_training_set)

    lr.train(training_labels)

    avg_in += lr.test(training_labels)

    lr.testing_set = cut_testing_set
    avg_out += lr.test(testing_labels)

    lr.testing_set = cut_validation_set
    avg_classification += lr.test(validation_labels)

    res = (avg_in, avg_out, avg_classification)
    return res

if __name__ == "__main__":
    question1 = Question("1. classification error",
                         [3, 4, 5, 6, 7], 'd')

    choices = [3, 4, 5, 6, 7]

    errors = [test_run(k) for k in choices]
    zipped = zip(choices, errors)

    answer1 = min(zipped, key=lambda c: c[1][2])
    question1.check(answer1[0])

    question2 = Question("2. out-of-sample error",
                         [3, 4, 5, 6, 7], 'e')

    answer2 = min(zipped, key=lambda c: c[1][1])
    question2.check(answer2[0])

    question3 = Question("3. classification for reverse training (10) and validation (25)",
                         [3, 4, 5, 6, 7], 'd')

    errors = [test_run(k, reverse=True) for k in choices]
    zipped = zip(choices, errors)

    answer3 = min(zipped, key=lambda c: c[1][2])
    question3.check(answer3[0])

    question4 = Question("4. classification error on 5 models (reversed)",
                         [3, 4, 5, 6, 7], 'd')

    answer4 = min(zipped, key=lambda c: c[1][1])
    question4.check(answer4[0])

    question5 = Question("5. out-of-sample for both models chosen by above 2 selections",
                         [(0.0, 0.1),
                          (0.1, 0.2),
                          (0.1, 0.3),
                          (0.2, 0.2),
                          (0.2, 0.3)], 'b',
                         lambda result, choice: abs(result[0] - choice[0]) + abs(result[1] - choice[1]))
    question5.check((answer1[1][1], answer3[1][1]))
