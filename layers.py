import numpy as np

from Neuron import Neuron
from activation_functions import ACTIVATION_FUNCTIONS
from exceptions import UnknownActivationFunctionError


class Layer:
    """
    Описывает объект слоя, являющийся общим для всех типов слоёв.
    """
    def __init__(self, neurons_count):
        self.neurons_count = neurons_count
        self.neurons = None

    def get_neurons_outputs(self):
        return np.array([neuron.output for neuron in self.neurons], dtype=np.float64)


class InputLayer(Layer):
    """
    Описывает объект входного слоя нейронной сети.
    """
    pass


class HiddenLayer(Layer):
    """
    Описывает объект скрытого слоя нейронной сети.
    """
    def __init__(self, neurons_count, activation_function_name):
        super().__init__(neurons_count)
        self.activation_function = ACTIVATION_FUNCTIONS.get(activation_function_name)

        if self.activation_function is None:
            raise UnknownActivationFunctionError


class OutputLayer(Layer):
    """
    Описывает объект выходного слоя нейронной сети.
    """
    def __init__(self, neurons_count, activation_function_name):
        super().__init__(neurons_count)
        self.activation_function = ACTIVATION_FUNCTIONS.get(activation_function_name)

        if self.activation_function is None:
            raise UnknownActivationFunctionError

    def set_gradients(self):
        if self.activation_function != 'softmax':
            raise 'Функция активации для выходного слоя не поддерживается.'

        dx_input = self.neurons[0].input
        dx_output = self.neurons[0].output
        # neurons_inputs = np.array([neuron.input for neuron in self.neurons])
        for neuron in self.neurons:
            if neuron.input == dx_input:
                neuron.local_gradient = neuron.output * (1 - neuron.output)
            else:
                neuron.local_gradient = -1 * neuron.output * dx_output

