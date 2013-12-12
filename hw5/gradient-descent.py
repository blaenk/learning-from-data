import sys
sys.path.append('..')

import numpy as np
import scipy
import random
import math

from common.model import Model
from common.question import Question

class GradientDescent(Model):
    def __init__(self, training_set=None, testing_set=None):
        Model.__init__(self, training_set, testing_set)

    def gradient(self):
        u, v = self.weights[0], self.weights[1]
        partial_u = 2 * (math.exp(v) + 2 * v * math.exp(-u)) * (u * math.exp(v) - 2 * v * math.exp(-u))
        partial_v = 2 * (math.exp(v) * u - 2 * v * math.exp(-u)) * (math.exp(v) * u - 2 * math.exp(-u))
        return np.array([partial_u, partial_v])

    def surface_error(self):
        u, v, = self.weights[0], self.weights[1]
        return (u * math.exp(v) - 2 * v * math.exp(-u)) ** 2

    def descend(self, start, coordinate=False):
        self.weights = start
        learning_rate = 0.1

        steps = 0
        tolerance = 10.0 ** -14

        print self.surface_error()

        if coordinate:
            condition = lambda: steps < 15
        else:
            condition = lambda: self.surface_error() > tolerance

        while condition():
            if coordinate:
                descent = self.gradient()
                self.weights[0] = self.weights[0] - learning_rate * descent[0]

                descent = self.gradient()
                self.weights[1] = self.weights[1] - learning_rate * descent[1]
            else:
                descent = self.gradient()
                self.weights = self.weights - descent * learning_rate

            steps += 1

        return steps

if __name__ == "__main__":
    question5 = Question("5. gradient descent number of iterations to meet threshold",
                         [1, 3, 5, 10, 17], 'd')
    question6 = Question("6. final (u, v)",
                         [(1., 1.),
                          (.713, .045),
                          (.016, .112),
                          (-.083, .029),
                          (.045, .024)], 'e',
                         lambda result, choice: abs(result[0] - choice[0]) + abs(result[1] - choice[1]))
    question7 = Question("7. coordinate descent error",
                         [10e-1, 10e-7, 10e-14, 10e-20], 'a',
                         lambda result, choice: 1 / abs(result - choice))

    gd = GradientDescent()

    iterations = gd.descend([1, 1])
    question5.check(iterations)

    uv = (gd.weights[0], gd.weights[1])
    question6.check(uv)

    gd.descend([1, 1], True)
    coordinate_error = gd.surface_error()
    question7.check(coordinate_error)
