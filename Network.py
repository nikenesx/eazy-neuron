import numpy as np

from layers import InputLayer, HiddenLayer, OutputLayer
from exceptions import FirstLayerError, HiddenLayerError, LastLayerError, UncorrectInputError
from Neuron import Neuron


class Network:
    def __init__(self, *layers):
        self._check_layers(layers)
        self.layers = layers
        self.lambda_s = 0.001

    def predict(self, v):
        vector = v
        if len(vector) != self.layers[0].neurons_count:
            raise UncorrectInputError

        # Задаем входные значения для нейронов входного слоя
        for i in range(len(vector)):
            self.layers[0].neurons[i].output = vector[i]

        # Вычисляем выходные значения для нейронов скрытых слоев
        for layer_index in range(1, len(self.layers) - 1):
            pred_layer_outputs = self.layers[layer_index - 1].get_neurons_outputs()
            for neuron in self.layers[layer_index].neurons:
                neuron.set_output(pred_layer_outputs)

        # Вычисляем входные значения для нейронов выходного слоя
        pred_layer_outputs = self.layers[-2].get_neurons_outputs()
        for neuron in self.layers[-1].neurons:
            neuron.input = np.dot(neuron.weights, pred_layer_outputs)
        last_layer_neurons_inputs = np.array([neuron.input for neuron in self.layers[-1].neurons])
        for neuron in self.layers[-1].neurons:
            neuron.output = neuron.activation_func(neuron.input, last_layer_neurons_inputs)

        return np.array([neuron.output for neuron in self.layers[-1].neurons])

    def configure(self):
        """
        Создаем нейроны для скрытых слоёв и задаем случайные веса для входных связей каждого нейрона.
        Количество входов для каждого нейрона - это количество нейронорв на предыдущем слое
        """
        for i in range(len(self.layers)):
            if i == 0:
                neurons = [Neuron() for _ in range(self.layers[i].neurons_count)]
                self.layers[i].neurons = neurons
                continue

            neurons = list()
            weights_count = self.layers[i - 1].neurons_count

            for _ in range(self.layers[i].neurons_count):
                weights = np.array([np.random.random() for _ in range(weights_count)])
                neuron = Neuron(weights, self.layers[i].activation_function)
                neurons.append(neuron)

            self.layers[i].neurons = neurons

    def correct_weights(self):
        for i in range(len(self.layers) - 1, 0, -1):
            for neuron in self.layers[i].neurons:
                for j in range(len(neuron.weights)):
                    neuron.weights[j] -= self.lambda_s * neuron.sum_gradients * self.layers[i - 1].neurons[j].output


    def train(self, input_data, result=None, batch_size=None, epochs=None, validation_split=None):
        """Обучение модели"""
        # if batch_size < 0 or epochs < 1 or validation_split < 0 or validation_split > 1:
        #     raise ValueError

        # Перемешиваем обучающие данные
        np.random.shuffle(input_data)

        iterations = 0

        # Проходим по всем обучающим матрицам
        for ind in range(len(input_data)):
            # Преобразуем матрицу в вектор
            vector = input_data[ind].flatten()
            if len(vector) != self.layers[0].neurons_count:
                raise UncorrectInputError

            # Задаем входные значения для нейронов входного слоя
            for i in range(len(vector)):
                self.layers[0].neurons[i].output = vector[i]

            # Вычисляем выходные значения для нейронов скрытых слоев
            for layer_index in range(1, len(self.layers) - 1):
                pred_layer_outputs = self.layers[layer_index - 1].get_neurons_outputs()
                for neuron in self.layers[layer_index].neurons:
                    neuron.set_output(pred_layer_outputs)

            # Вычисляем входные значения для нейронов выходного слоя
            pred_layer_outputs = self.layers[-2].get_neurons_outputs()
            for neuron in self.layers[-1].neurons:
                neuron.input = np.dot(neuron.weights, pred_layer_outputs)
            last_layer_neurons_inputs = np.array([neuron.input for neuron in self.layers[-1].neurons])
            for neuron in self.layers[-1].neurons:
                neuron.output = neuron.activation_func(neuron.input, last_layer_neurons_inputs)

            output_vector = np.array([neuron.output for neuron in self.layers[-1].neurons])
            e_vector = output_vector - result[ind].flatten()

            # <======================================== Выходной слой ========================================>
            dx_input = self.layers[-1].neurons[0].input
            dx_output = self.layers[-1].neurons[0].output
            for i in range(len(self.layers[-1].neurons)):
                neuron = self.layers[-1].neurons[i]
                if neuron.input == dx_input:
                    neuron.local_gradient = neuron.output * (1 - neuron.output) * e_vector[i]
                else:
                    neuron.local_gradient = -1 * neuron.output * dx_output * e_vector[i]
                neuron.sum_gradients += neuron.local_gradient

            # </================================================================================================>

            for i in range(len(self.layers) - 2, 0, -1):
                current_layer = self.layers[i]
                pred_layer = self.layers[i + 1]
                for j in range(len(current_layer.neurons)):
                    sum_gradients = 0
                    for k in range(len(pred_layer.neurons)):
                        sum_gradients += pred_layer.neurons[k].weights[j] * pred_layer.neurons[k].local_gradient

                    df = 1 if current_layer.neurons[j].output > 0 else 0
                    current_layer.neurons[j].local_gradient = sum_gradients * df
                    current_layer.neurons[j].sum_gradients += current_layer.neurons[j].local_gradient

            iterations += 1
            if iterations == batch_size:
                self.correct_weights()
                iterations = 0

    @staticmethod
    def _check_layers(layers):
        """
        Проверяем, что первый слой это объект типа InputLayer, последний слой объект типа OutputLayer и скрытые слои
        объекты типа HiddenLayer.
        """
        if len(layers) < 2:
            raise 'Ты даун?'

        if not isinstance(layers[0], InputLayer):
            raise FirstLayerError

        if not isinstance(layers[-1], OutputLayer):
            raise LastLayerError

        for layer in layers[1:-1]:
            if not isinstance(layer, HiddenLayer):
                raise HiddenLayerError
