from tensorflow import keras
from tensorflow.keras import layers


class experiment:
    def __init__(self, config, strategy):
        self.config = config
        self.strategy = strategy
        self.model = self.__build__()

    def __build__(self) -> keras.Model:
        pass
