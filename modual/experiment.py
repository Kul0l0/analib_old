#!/usr/bin/env python

from tensorflow import keras
from tensorflow.keras import layers
from . import block
import os


class experiment:
    def __init__(self, *, model_config, strategy=None):
        """
        define a experiment with a modual and train strategy
        :param model_config: dict
            required key:
                input_shape: int, the image shape for inputting the model
                model_config: tuple, the struct of model

        :param strategy:
        """
        self.model_config = model_config
        self.strategy = strategy
        self.code = model_config.get('code', block.random_str(3))
        self.name = '%s_%s' % (model_config.get('name', 'model'), self.code)
        self.model = self.__build__()
        if self.model_config.get('plot'):
            keras.utils.plot_model(
                model=self.model,
                to_file='%s.png' % os.path.join(self.model_config.get('outpath', './'), self.name),
                show_shapes=True,
            )
        # load strategy
        if strategy:
            self.cv = strategy.pop('cv', None)
            self.batch_size = strategy.pop('batch_size')
            self.epochs = strategy.pop('epochs')

    def __build__(self) -> keras.Model:
        # define input and output block
        input_shape = self.model_config.get('input_shape')
        inputs = keras.Input(shape=(input_shape, input_shape, 3))
        # build model structure
        outputs = build_block(inputs, self.model_config.get('model_config'))
        return keras.Model(inputs=inputs, outputs=outputs, name=self.name)

    def __train__(self, train_dataset, val_dataset=None):
        pass


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
def build_block(features, config):
    # type(config)==tuple: connect
    # type(config)==list: parallel
    if isinstance(config, list):
        branch_outputs = list()
        for sub_config in config:
            branch_outputs.append(build_block(features, sub_config))
        return layers.Concatenate()(branch_outputs)
    elif isinstance(config, tuple):
        for sub_config in config:
            features = build_block(features, sub_config)
        return features
    elif isinstance(config, dict):
        times, name, args = config['times'], config['name'], config['args']
        for i in range(times):
            if name == 'TF':
                features = args(features)
            else:
                keymaping(args)
                features = block.BLOCK_MAP[name](**args)(features)
        return features
    else:
        raise TypeError("Error Type of config")
