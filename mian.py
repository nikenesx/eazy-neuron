import numpy as np

from Network import Network
from layers import InputLayer, HiddenLayer, OutputLayer
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

MODEL_NAME = 'digit_recognizer.h5'


def train():
    # Загружаем обучающие данные
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Нормализуем обучающие данные
    x_train = x_train / 255

    model = Network(
        InputLayer(784),
        HiddenLayer(10, 'sigmoid'),
        OutputLayer(10, 'softmax')
    )
    model.configure()
    model.train(x_train, y_train, batch_size=32)

    x = np.expand_dims(x_test[0], axis=0)
    a = model.predict(x)
    print(a)

    plt.imshow(x_test[0], cmap=plt.cm.binary)
    plt.show()


if __name__ == '__main__':
    train()
