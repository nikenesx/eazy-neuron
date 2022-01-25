import numpy as np


def sigmoid(x):
    """Сигмоидальная функция активации"""
    return np.float16(1 / (1 + np.exp(-x)))


def relu(x):
    """Функция акцивации ReLu"""
    return np.float64(x if x > 0 else 0)


def softmax(neurons_inputs):
    """
    Функция активации softmax

    :param neurons_inputs: список с входными значениями остальных нейронов текущего слоя
    :return: выходное значение нейрона
    """
    return np.float16(np.exp(neurons_inputs) / np.sum(neurons_inputs))


ACTIVATION_FUNCTIONS = {
    'sigmoid': sigmoid,
    'relu': relu,
    'softmax': softmax,
}
