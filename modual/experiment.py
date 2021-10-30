from tensorflow import keras
from tensorflow.keras import layers
import block


def hw():
    print("hw")


class experiment:
    def __init__(self, *, config, strategy):
        """
        define a experiment with a modual and train strategy
        :param config: dict
            required key:
                input_shape: int, the image shape for inputting the model
                model_config: tuple, the struct of model

        :param strategy:
        """
        self.config = config
        self.strategy = strategy
        self.model = self.__build__()

    def __build__(self) -> keras.Mondel:
        # map shortcut to fullname
        def keymaping(args):
            keymap = {
                'k': 'kernel_size',
                'f': 'filters',
                's': 'strides',
                'p': 'padding',
                'a': 'activation',
            }
            for k, v in keymap.items():
                if k in args:
                    args[v] = args.pop(k)

        # build block
        def build_block(block_inputs, config):
            times, name, args = config['times'], config['name'], config['args']
            keymaping(args)
            for i in range(times):
                block_inputs = block.BLOCK_MAP[name](**args)(block_inputs)
            return block_inputs

        # define input and output block
        input_shape = self.config.get('input_shape')
        inputs = keras.Input(shape=(input_shape, input_shape, 3))
        outputs = inputs
        # build model structure
        for block_config in self.config.get('model_config'):
            # tree structure
            if isinstance(block_config, tuple):
                branch_outputs = list()
                for sub_block_config in block_config:
                    branch_outputs.append(build_block(outputs, sub_block_config))
                outputs = layers.concatenate()(branch_outputs)
            elif isinstance(block_config, dict):
                outputs = build_block(outputs, block_config)
            else:
                raise TypeError("Error Type of block_config")

        return keras.Model(inputs=inputs, outputs=outputs)
