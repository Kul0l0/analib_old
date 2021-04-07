#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : Kul0l0
@File    : cnn.py
@Time    : 2021/3/8 下午3:16
'''
from tensorflow import keras
from tensorflow.keras import Sequential, layers, Model
from tensorflow.keras.layers import Layer
import const


def CB_block(input = None, input_shape: set = (256,256,3), filters: int = 3, kernel_size: int = 3, strides: int = 1, padding: object = 'same'):
    if input is None:
        input = keras.Input(shape=input_shape)
    if padding in ('same', 'valid'):
        output = (
            layers.Conv2D(
                filters, kernel_size, strides, padding=padding,
                kernel_regularizer=keras.regularizers.L2(l2=0.0001),
                bias_regularizer=keras.regularizers.L2(l2=0.0001),
            )
        )(input)
    else:
        output = layers.ZeroPadding2D(padding=padding)(input)
        output = (
            layers.Conv2D(
                filters, kernel_size, strides, padding='valid',
                kernel_regularizer=keras.regularizers.L2(l2=0.0001),
                bias_regularizer=keras.regularizers.L2(l2=0.0001)
            )
        )(output)
    output = layers.BatchNormalization()(output)
    return output


def CBA_block(input = None, input_shape: set = (256,256,3), filters: int = 3, kernel_size: int = 3, strides: int = 1, padding: object = 'same'):
    output = CB_block(input, input_shape, filters, kernel_size, strides, padding)
    output = layers.ReLU()(output)
    return output

class ResCommonBlock(Layer):
    def __init__(self, begin, filters:int, kernel_size:int=3, strides:int=1):
        super(ResCommonBlock, self).__init__()
        self.begin = begin
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.shortcut = None
        if begin or strides==2:
            self.shortcut = CB_block(self.filters, kernel_size=1, strides=self.strides, padding='valid')
        self.block = self.install(begin, filters, kernel_size, strides)

    def install(self, begin, filters, kernel_size, strides):
        res = Sequential()
        res.add(CBA_block(filters, kernel_size, strides=strides, padding='same'))
        res.add(CB_block(filters, kernel_size, strides=1, padding='same'))
        return res

    def call(self, inputs, **kwargs):
        if self.shortcut is None:
            outputs = layers.add([self.block(inputs), inputs])
        else:
            outputs = layers.add([self.block(inputs), self.shortcut(inputs)])
        return layers.ReLU()(outputs)

class ResBottleneckBlock(Layer):
    def __init__(self, begin, filters:int, kernel_size:int=3, strides:int=1):
        super(ResBottleneckBlock, self).__init__()
        self.begin = begin
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.shortcut = None
        if begin or strides==2:
            self.shortcut = CB_block(self.filters*4, kernel_size=1, strides=self.strides, padding='valid')
        self.block = self.install(begin, filters, kernel_size, strides)

    def install(self, begin, filters, kernel_size, strides):
        res = Sequential()
        res.add(CBA_block(filters, kernel_size=1, strides=strides, padding='valid'))
        res.add(CBA_block(filters, kernel_size, strides=1, padding='same'))
        res.add(CB_block(filters*4, kernel_size=1, strides=1, padding='valid'))
        return res

    def call(self, inputs, **kwargs):
        if self.shortcut is None:
            outputs = layers.add([self.block(inputs), inputs])
        else:
            outputs = layers.add([self.block(inputs), self.shortcut(inputs)])
        return layers.ReLU()(outputs)

# block_dict
BLOCK_MAP = {
    'CB': CB_block,
    'CBA': CBA_block,
}

def build(input_shape, block_list: list, config_list: list):
    '''
    :param config:
    input_shape: a set, example: (256, 256, 3)
    block_type: str or list of block type
    block_config: list of dict
    :type config: dict
    :return:
    :rtype: tf model
    '''
    input, output = keras.Input(shape=input_shape), None
    for block, config in zip(block_list, config_list):
        if isinstance(block, str):
            times = 1
        else:
            block, times = block
        for i in range(times):
            output = BLOCK_MAP[block](**config)
    return keras.Model(inputs=input, outputs=output)




def block_list():
    print(list(BLOCK_MAP.keys()))

class Resnet(Model):
    def __init__(self, data_set, block_type, img_size, class_number, stack_list, filter_list):
        super(Resnet, self).__init__()
        self.data_set = data_set
        self.block_type = block_type
        self.img_size = img_size
        self.class_number = class_number
        self.stack_list = stack_list
        self.filter_list = filter_list
        self.head = self.install_head()
        self.body = self.install_body()
        self.tail = self.install_tail()

    def call(self, inputs, training=None, mask=None):
        outputs = self.head(inputs)
        outputs = self.body(outputs)
        return self.tail(outputs)

    def plain_block(self, begin, filters: int, kernel_size:int=3, strides:int=1):
        res = Sequential()
        res.add(CBA_block(filters, kernel_size, strides, padding='same'))
        res.add(CBA_block(filters, kernel_size, strides=1, padding='same'))
        return res

    def common_block(self, begin, filters:int, kernel_size:int=3, strides:int=1):
        res = Sequential()
        res.add(ResCommonBlock(begin, filters, kernel_size, strides))
        return res

    def bottleneck_block(self, begin, filters:int, kernel_size:int=3, strides:int=1):
        res = Sequential()
        res.add(ResBottleneckBlock(begin, filters, kernel_size, strides))
        return res

    def install_head(self):
        res = Sequential()
        if self.data_set == 'cifar10':
            res.add(layers.Conv2D(16, 3, 1, 'same', activation='relu', kernel_regularizer=keras.regularizers.L2(l2=0.0001), bias_regularizer=keras.regularizers.L2(l2=0.0001)))
        else: # imagenet
            res.add(self.CBA_block(64, 7, 2, padding=7))
            res.add(layers.MaxPool2D(pool_size=3, strides=2, padding=((1,1),(1,1))))
        return res

    def install_body(self):
        block_map = {
            'plain': self.plain_block,
            'common': self.common_block,
            'bottleneck': self.bottleneck_block,
        }
        block = block_map[self.block_type]
        res = Sequential()
        for idx, (filters, stack) in enumerate(zip(self.filter_list, self.stack_list)):
            for i in range(stack):
                if i == 0:
                    if idx == 0:
                        res.add(block(begin=True, filters=filters, kernel_size=3, strides=1))
                    else:
                        res.add(block(begin=False, filters=filters, kernel_size=3, strides=2))
                else:
                    res.add(block(begin=False, filters=filters, kernel_size=3, strides=1))
        return res

    def install_tail(self):
        res = Sequential()
        res.add(layers.GlobalAveragePooling2D())
        res.add(layers.Dense(self.class_number, activation='softmax', kernel_regularizer=keras.regularizers.L2(l2=0.0001)))
        return res
