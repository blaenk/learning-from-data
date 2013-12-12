import sys
sys.path.append('..')

import numpy as np
import scipy
import random

from common.model import Model
from common.question import Question

from hw1.perceptron import Perceptron

class LinearRegression(Model):
    def __init__(self, training_set=None, testing_set=None):
        Model.__init__(self, training_set, testing_set)
        self.weights = np.array([0., 0., 0.])
        # points that define the target function
        self.point1 = (random.uniform(-1, 1), random.uniform(-1, 1))
        self.point2 = (random.uniform(-1, 1), random.uniform(-1, 1))

    def target(self, feature):
        x = feature[1]
        y = feature[2]
        x1, y1 = self.point1
        x2, y2 = self.point2
        slope = (y2 - y1) / (x2 - x1)
        # simple check to see if point (x, y) is above or below the line
        return 1 if y > (slope * (x - x1) + y1) else -1

    def train(self, noise=None):
        data_matrix = np.array(self.training_set)
        target_vector = np.array([[self.target(point)] for point in self.training_set])

        if noise is not None:
          self.add_noise(target_vector, noise)

        # normal equations linear regression
        pinv = np.linalg.pinv(data_matrix)
        self.weights = np.dot(pinv, target_vector)

    def add_noise(self, targets, noise):
        noisy_points = set()
        num_points = len(self.testing_set)
        cap = int(num_points * noise)

        while len(noisy_points) < cap:
          index = random.randint(0, num_points - 1)
          if not index in noisy_points:
            targets[index][0] *= -1
            noisy_points.add(index)

    def test(self, noise=None):
        targets = np.array([[self.target(point)] for point in self.testing_set])

        if noise is not None:
          self.add_noise(targets, noise)

        tys = np.sign(np.dot(self.testing_set, self.weights))
        return len(targets[tys != targets]) / float(len(targets))

def test_run(test_runs, in_sample, perceptron, training_size=100, testing_size=1000):
    avg_error = 0

    for i in xrange(test_runs):
        training_set = [[1, random.uniform(-1, 1), random.uniform(-1, 1)]
                        for i in xrange(training_size)]
        lr = LinearRegression(training_set)

        if in_sample:
          lr.testing_set = training_set
        else:
          lr.testing_set = [[1, random.uniform(-1, 1), random.uniform(-1, 1)]
                            for i in xrange(testing_size)]

        lr.train()

        if perceptron:
          pla = Perceptron(training_set=training_set, weights=lr.weights)

          # this is actually the number of iterations
          avg_error += pla.train()
        else:
          avg_error += lr.test()

    avg_error /= float(test_runs)
    return avg_error

if __name__ == "__main__":
    question5 = Question("5. [n = 100] in sample error",
                         [0, 0.001, 0.01, 0.1, 0.5], 'c')

    question6 = Question("6. [n = 1000] out of sample error",
                         [0, 0.001, 0.01, 0.1, 0.5], 'c')

    question7 = Question("7. [n = 10] avg. iterations for perceptron with lin. reg. weights",
                         [1, 15, 300, 5000, 10000], 'a')

    in_sample = test_run(1000, True, False)
    question5.check(in_sample)

    # 5. [n = 100] in sample error
    #   result: 0.04071    nearest: c. 0.01      answer: c. 0.01    CORRECT

    out_of_sample = test_run(1000, False, False)
    question6.check(out_of_sample)

    # 6. [n = 1000] out of sample error
    #   result: 0.048075   nearest: c. 0.01      answer: c. 0.01    CORRECT

    iterations = test_run(1000, True, True, training_size=10)
    question7.check(1)

    # 7. [n = 10] avg. iterations for perceptron with lin. reg. weights
    #   result: 1          nearest: a. 1         answer: a. 1       CORRECT

