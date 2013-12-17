import sys
sys.path.append('..')

import numpy as np
import scipy
import random
import math

from common.model import Model
from common.question import Question


class LogisticRegression(Model):
    def __init__(self, training_set=None, testing_set=None):
        Model.__init__(self, training_set, testing_set)

        # points that define the target function
        self.point1 = (random.uniform(-1, 1), random.uniform(-1, 1))
        self.point2 = (random.uniform(-1, 1), random.uniform(-1, 1))
        self.weights = [0., 0., 0.]

    def target(self, feature):
        x, y = feature[1], feature[2]
        x1, y1 = self.point1
        x2, y2 = self.point2
        slope = (y2 - y1) / (x2 - x1)
        # simple check to see if point (x, y) is above or below the line
        return 1 if y > (slope * (x - x1) + y1) else -1

    def cross_entropy(self, feature, result):
        return math.log(1 + math.exp(-result * np.dot(feature, self.weights)))

    def gradient(self, point, result):
        num = -np.multiply(result, point)
        den = 1 + (math.exp(result * np.dot(self.weights, point)))
        return np.divide(num, den)

    def train(self):
        target_vector = np.array([[self.target(point)]
                                  for point in self.training_set])

        learning_rate = 0.01
        previous_weights = [-1, -1, -1]
        step = 0
        displacement = 1

        sample_size = len(target_vector)

        while not displacement < 0.01:
            permuted_order = range(sample_size)
            random.shuffle(permuted_order)

            for idx in permuted_order:
              feature = self.training_set[idx]
              result = target_vector[idx]

              descent = self.gradient(feature, result)
              self.weights = self.weights - learning_rate * descent

            displacement = np.linalg.norm(previous_weights - self.weights)
            previous_weights = self.weights

            step += 1

        return step

    def test(self):
        cross_entropy = 0.0

        for feature in self.testing_set:
            result = self.target(feature)
            cross_entropy += self.cross_entropy(feature, result)

        return cross_entropy / 1000

def test_run():
    steps = 0.0
    error = 0.0

    for i in xrange(100):
        print i
        training_set = [[1., random.uniform(-1, 1), random.uniform(-1, 1)] for i in xrange(100)]
        testing_set = [[1., random.uniform(-1, 1), random.uniform(-1, 1)] for i in xrange(1000)]
        lg = LogisticRegression(training_set, testing_set)
        steps += lg.train()
        error += lg.test()

    steps /= 100
    error /= 100
    return steps, error

if __name__ == "__main__":
    question8 = Question("8. [n = 100] out of sample error",
                         [.025, .050, .075, .1, .125], 'd')
    question9 = Question("9. [n = 100] epochs to converge",
                         [350, 550, 750, 950, 1750], 'a')

    steps, error = test_run()
    question8.check(error)
    question9.check(steps)
