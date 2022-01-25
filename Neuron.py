import numpy as np


class Neuron:
    def __init__(self, weights=None, activation_func=None):
        self.weights = weights
        self.activation_func = activation_func
        self.local_gradient = None
        self.sum_gradients = 0
        self.input = None
        self.output = None

    def set_output(self, inputs, *args):
        """Выходное значение нейрона"""
        self.input = np.dot(self.weights, inputs)
        self.output = self.activation_func(self.input, *args)

