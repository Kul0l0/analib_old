#!/usr/bin/env python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split, KFold
from . import block, metrics
import os


class experiment:
    def __init__(self, *, experiment_config, model_config, strategy=None, fit_config=None, metrics=None):
        """
        define a experiment with a modual and train strategy
        :param model_config: dict
            required key:
                input_shape: int, the image shape for inputting the model
                model_config: tuple, the struct of model

        :param strategy:
        """
        # experiment_config
        self.ep_name = experiment_config.get('name')
        self.outdir = experiment_config.get('outdir')
        self.label_name = experiment_config.get('label_name')
        # build output dir
        if self.outdir is None:
            os.makedirs('./%s' % self.ep_name, exist_ok=True)
        else:
            os.makedirs(self.outdir, exist_ok=True)
        # model_config
        self.model_config = model_config
        self.model_code = model_config.get('code', block.random_str(3))
        self.model_name = '%s_%s' % (model_config.get('name', 'model'), self.model_code)
        # build model
        self.__build__()
        if self.model_config.get('plot'):
            keras.utils.plot_model(
                model=self.model,
                to_file='%s_structure.png' % os.path.join(self.outdir, self.model_name),
                show_shapes=True,
            )
        # load strategy
        if strategy:
            self.kfold = strategy.pop('kfold', None)
            self.seed = strategy.pop('seed', None)
            self.strategy = strategy
        # fit_config
        if fit_config:
            self.fit_config = fit_config
            self.y_true = np.array([])
            self.y_pred = np.array([])
        # metrics
        self.metrics = metrics

    def __build__(self):
        # define input and output block
        input_shape = self.model_config.get('input_shape')
        if isinstance(input_shape, int):
            inputs = keras.Input(shape=(input_shape, input_shape, 3))
        else:
            # input_shape is tuple
            inputs = keras.Input(shape=input_shape)
        outputs = build_block(inputs, self.model_config.get('model_structure'))
        # build model
        self.model = keras.Model(inputs=inputs, outputs=outputs, name=self.model_name)

    def __compile__(self):
        self.model.compile(**self.strategy)

    def __fit__(self, train_dataset, val_dataset):
        self.fit_config['x'] = train_dataset
        self.fit_config['validation_data'] = val_dataset
        self.model.fit(**self.fit_config)

    def __metrics__(self, y_true, y_pred, outdir):
        if 'confusion_matrix' in self.metrics:
            metrics.confusion_matrix(
                y_true=y_true,
                y_pred=y_pred,
                label_name=self.label_name,
                outdir=outdir,
            )

    def __get_dataset__(self, data, augment=None, val=None):
        batch_size = self.fit_config.get('batch_size')
        if val:
            return get_dataset(data, batch_size)
        return get_dataset(data, batch_size, augment)

    def train(self, data, augment=None, test_size=None):
        if self.kfold:
            x, y = data
            kf = KFold(self.kfold, random_state=self.seed, shuffle=True)
            counter = 1
            for train_idx, val_idx in kf.split(x):
                # mkdir
                model_outdir = os.path.join(self.outdir, "%s_%d" % (self.model_name, counter))
                os.makedirs(model_outdir, exist_ok=True)
                # get dataset
                train_data = x[train_idx], y[train_idx]
                val_data = x[val_idx], y[val_idx]
                train_dataset = self.__get_dataset__(train_data, augment=augment)
                val_dataset = self.__get_dataset__(val_data, val=True)
                # counter
                print("### KFold: %s/%s ###" % (counter, self.kfold))
                counter += 1
                # compile model
                self.__build__()
                self.__compile__()
                self.__fit__(train_dataset, val_dataset)
                # save result
                sub_y_pred = np.argmax(self.model.predict(x[val_idx]), axis=1)
                sub_y_true = y[val_idx]
                self.__metrics__(sub_y_true, sub_y_pred, outdir=model_outdir)
                self.y_pred = np.append(self.y_pred, sub_y_pred)
                self.y_true = np.append(self.y_true, sub_y_true)
            self.__metrics__(self.y_true, self.y_pred, outdir=self.outdir)
        else:
            # get dataset
            train_data, val_data = split_data(data, test_size, random_state=self.seed)
            train_dataset = self.__get_dataset__(train_data, augment=augment)
            val_dataset = self.__get_dataset__(val_data, val=True)
            # compile model
            self.__compile__()
            self.__fit__(train_dataset, val_dataset)
            self.__metrics__(self.y_true, self.y_pred, outdir=self.outdir)


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


def split_data(dataset, test_size=None, random_state=369):
    if test_size is None:
        return dataset, None
    else:
        train_x, val_x, train_y, val_y = train_test_split(*dataset, test_size=test_size, random_state=random_state)
        return (train_x, train_y), (val_x, val_y)


def get_dataset(data, batch_size, augment=None, val=False) -> tf.data.Dataset:
    """
    :param data: tuple, (data, label)
    :param batch_size:
    :param augment:
    :param val:
    :return:
    """
    dataset = tf.data.Dataset.from_tensor_slices(data)
    if val is True:
        return dataset.batch(batch_size=batch_size)
    else:
        dataset = dataset.shuffle(buffer_size=batch_size*3).batch(batch_size=batch_size, drop_remainder=True)
        if augment:
            dataset = dataset.map(augment, num_parallel_calls=-1)
        return dataset.prefetch(-1)
