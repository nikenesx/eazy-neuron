class UnknownActivationFunctionError(Exception):
    def __str__(self):
        return 'Неизвестная ункция активации'


class FirstLayerError(Exception):
    def __str__(self):
        return 'Первый слой должен быть объектом типа InputLayer.'


class LastLayerError(Exception):
    def __str__(self):
        return 'Последний слой должен быть объектом типа OutputLayer.'


class HiddenLayerError(Exception):
    def __str__(self):
        return 'Скрытый слой должен быть объектом типа HiddenLayer.'


class UncorrectInputError(Exception):
    def __str__(self):
        return 'Длина входного вектора не соответствует количеству нейронов входного слоя.'


class NotConfiguredNetwork(Exception):
    def __str__(self):
        return 'Сеть не была сконфигурирована.'
