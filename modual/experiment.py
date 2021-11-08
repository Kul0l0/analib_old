#!/usr/bin/env python

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from . import block
import os


class experiment:
    def __init__(self, *, model_config, strategy=None, fit=None):
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
            self.strategy = strategy
        if fit:
            self.fit = fit

    def __build__(self) -> Model:
        # define input and output block
        input_shape = self.model_config.get('input_shape')
        inputs = keras.Input(shape=(input_shape, input_shape, 3))
        # build model structure
        outputs = build_block(inputs, self.model_config.get('model_config'))
        return keras.Model(inputs=inputs, outputs=outputs, name=self.name)

    def __compile__(self):
        self.model.compile(**self.strategy)

    def train(self, data, augment=None, test_size=None, random_state=369):
        if self.cv:
            pass
        else:
            # get data
            train_data, val_data = split_dataset(data, test_size, random_state=random_state)
            batch_size = self.fit.get('batch_size')
            train_dataset = get_dataset(train_data, batch_size=batch_size, augment=augment)
            val_dataset = get_dataset(val_data, batch_size=batch_size, val=True)
            # compile model
            self.__compile__()
            self.fit['x'] = train_dataset
            self.fit['validation_data'] = val_dataset
            self.model.fit(**self.fit)


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


def split_dataset(dataset, test_size=None, random_state=369):
    if test_size is None:
        return dataset, None
    else:
        train_x, val_x, train_y, val_y = train_test_split(*dataset, test_size=test_size, random_state=random_state)
        return (train_x, train_y), (val_x, val_y)


def get_dataset(dataset, batch_size, augment=None, val=False) -> tf.data.Dataset:
    """
    :param dataset: tuple, (data, label)
    :param batch_size:
    :param augment:
    :param val:
    :return:
    """
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    if val is True:
        return dataset.batch(batch_size=batch_size)
    else:
        dataset = dataset.shuffle(buffer_size=batch_size*3).batch(batch_size=batch_size, drop_remainder=True)
        if augment:
            dataset = dataset.map(augment, num_parallel_calls=-1)
        return dataset.prefetch(-1)
