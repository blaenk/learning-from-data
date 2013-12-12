import sys
sys.path.append('..')

import numpy as np
import scipy
import random

from common.model import Model
from common.question import Question

from linear_regression import LinearRegression

def test_run(in_sample, complex):
    test_runs = 1000
    training_size = testing_size = 1000
    avg_error = 0
    q9 = None # will store dataset and weights

    for i in xrange(test_runs):
        training_set = [[1, random.uniform(-1, 1), random.uniform(-1, 1)]
                        for i in xrange(training_size)]

        # non-linear transformation
        if complex:
            transform = lambda feature: [1, feature[1], feature[2], feature[1] * feature[2], feature[1] ** 2, feature[2] ** 2]
            training_set = [transform(feature) for feature in training_set]

        lr = LinearRegression(training_set)
        lr.target = lambda feature: np.sign(feature[1] ** 2 + feature[2] ** 2 - 0.6)

        if in_sample:
          lr.testing_set = training_set
        else:
          lr.testing_set = [[1, random.uniform(-1, 1), random.uniform(-1, 1)]
                            for i in xrange(testing_size)]

          if complex:
            lr.testing_set = [transform(feature) for feature in lr.testing_set]

        if not complex:
            lr.train()
        else:
            lr.train(noise=0.10)

        if q9 is None:
            q9 = (lr.testing_set, lr.weights)

        if not complex:
            avg_error += lr.test()
        else:
            avg_error += lr.test(noise=0.10)

    avg_error /= float(test_runs)
    return avg_error, q9

def q9_hypothesis(choice, x1, x2):
  return np.sign(choice[0] +
                 choice[1] * x1 +
                 choice[2] * x2 +
                 choice[3] * x1 * x2 +
                 choice[4] * x1 ** 2 +
                 choice[5] * x2 ** 2)

def hypothesis_score(data):
  # the question says that the closest choice is the one that agrees
  # most with our own hypothesis (result): h(x) == g(x)

  # we generate the target ys using the sign(X * w) formula
  # then for every feature in the data set we generate the targets
  # using a particular choice hypothesis

  # the score of any particular choice hypothesis is the average
  # agreeability: the number of mismatched targets / total number of targets
  def match(result, choice):
    mat = np.array(data)

    # generate results from our weights
    hy = np.sign(np.dot(mat, result)).T[0]

    # vectorize the choice hypothesis functions
    g = np.vectorize(q9_hypothesis, excluded=['choice'])

    # generate the targets
    gy = g(choice=choice, x1=mat[:, 1], x2=mat[:, 2])

    # calculate the average agreeability
    return len(hy[gy != hy]) / float(len(hy))

  return match

if __name__ == "__main__":
    question8 = Question("8. in sample error",
                         [0, 0.1, 0.3, 0.5, 0.8], 'd')

    in_sample, _ = test_run(True, False)
    question8.check(in_sample)

    # 8. in sample error
    #   result: 0.507416   nearest: d. 0.5       answer: d. 0.5     CORRECT

    out_sample, q9 = test_run(True, True)
    data, weights = q9

    question9 = Question("9. hypothesis",
                         [[-1, -0.5, 0.08, 0.13, 1.50, 1.50],
                          [-1, -0.5, 0.08, 0.13, 1.50, 15.0],
                          [-1, -0.5, 0.08, 0.13, 15.0, 1.50],
                          [-1, -1.5, 0.08, 0.13, 0.05, 0.05],
                          [-1, -0.5, 0.08, 1.50, 0.15, 0.15]], 'a',
                         hypothesis_score(data))

    question9.check(weights)

    # 9. hypothesis
    #   result: [-1.00796613 -0.02536106  0.01250888 -0.05369976  1.56308802  1.57828083]
    #   nearest: a. [-1, -0.5, 0.08, 0.13, 1.5, 1.5]
    #   answer:  a. [-1, -0.5, 0.08, 0.13, 1.5, 1.5] CORRECT

    question10 = Question("10. out of sample error",
                          [0, 0.1, 0.3, 0.5, 0.8], 'b')
    question10.check(out_sample)

    # 10. out of sample error
    #   result: 0.125085   nearest: b. 0.1       answer: b. 0.1     CORRECT
