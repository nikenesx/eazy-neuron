from math import exp
from mpmath import mp


def sigmoid(x):
    """Сигмоидальная функция активации"""
    return 1 / (1 + mp.exp(-x))


def relu(x):
    """Функция акцивации ReLu"""
    return x if x > 0 else 0


def softmax(x, neurons_inputs):
    """
    Функция активации softmax

    :param x: входное значение текущего нейрона
    :param neurons_inputs: список с входными значениями остальных нейронов текущего слоя
    :return: выходное значение нейрона
    """
    return mp.exp(x) / sum([mp.exp(value) for value in neurons_inputs])


ACTIVATION_FUNCTIONS = {
    'sigmoid': sigmoid,
    'relu': relu,
    'softmax': softmax,
}
