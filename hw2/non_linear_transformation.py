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
    weights = None

    for i in xrange(test_runs):
        training_set = [[1, random.uniform(-1, 1), random.uniform(-1, 1)]
                        for i in xrange(training_size)]

        # non-linear transformation
        if complex:
            transform = lambda x, y: [1, x, y, x * y, x ** 2, y ** 2]
            training_set = [transform(point[1], point[2]) for point in training_set]

        lr = LinearRegression(training_set)
        lr.target = lambda point: 1 if (point[1] ** 2 + point[2] ** 2 - 0.6) > 0 else -1

        if in_sample:
          lr.testing_set = training_set
        else:
          lr.testing_set = [[1, random.uniform(-1, 1), random.uniform(-1, 1)]
                            for i in xrange(testing_size)]

          if complex:
            lr.testing_set = [transform(point[1], point[2]) for point in lr.testing_set]

        if not complex:
            lr.train()
        else:
            lr.train(noise=0.10)

        if weights is None:
            weights = lr.weights

        if not complex:
            avg_error += lr.test()
        else:
            avg_error += lr.test(noise=0.10)

    avg_error /= float(test_runs)
    return avg_error, weights

if __name__ == "__main__":
    question8 = Question("8. in sample error",
                         [0, 0.1, 0.3, 0.5, 0.8], 'd')

    in_sample, _ = test_run(True, False)
    question8.check(in_sample)

    # 8. in sample error
    #   result: 0.507416   nearest: d. 0.5       answer: d. 0.5     CORRECT

    question9 = Question("9. hypothesis",
                         [[-1, -0.5, 0.08, 0.13, 1.50, 1.50],
                          [-1, -0.5, 0.08, 0.13, 1.50, 15.0],
                          [-1, -0.5, 0.08, 0.13, 15.0, 1.50],
                          [-1, -1.5, 0.08, 0.13, 0.05, 0.05],
                          [-1, -0.5, 0.08, 1.50, 0.15, 0.15]], 'a',
                         # polynomial scoring lambda
                         lambda result, choice:
                         sum([abs(result_nomial - nomial)
                             for result_nomial, nomial
                             in zip(result, choice)]))

    out_sample, weights = test_run(True, True)
    question9.check(weights.T[0])

    # 9. hypothesis
    #   result: [-1.00796613 -0.02536106  0.01250888 -0.05369976  1.56308802  1.57828083]
    #   nearest: a. [-1, -0.5, 0.08, 0.13, 1.5, 1.5]
    #   answer:  a. [-1, -0.5, 0.08, 0.13, 1.5, 1.5] CORRECT

    question10 = Question("10. out of sample error",
                          [0, 0.1, 0.3, 0.5, 0.8], 'b')
    question10.check(out_sample)

    # 10. out of sample error
    #   result: 0.125085   nearest: b. 0.1       answer: b. 0.1     CORRECT
