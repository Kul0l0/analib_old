#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : Kul0l0
@File    : cnn.py
@Time    : 2021/3/8 下午3:16
'''
from tensorflow.keras import Sequential, layers, Model
from tensorflow.keras.layers import Layer


def CB_block(filters: int, kernel_size: int = 3, strides: int = 1, padding: object = 'same'):
    res = Sequential()
    if padding in ('same', 'valid'):
        res.add(layers.Conv2D(filters, kernel_size, strides, padding=padding))
    else:
        res.add(layers.ZeroPadding2D(padding=padding))
        res.add(layers.Conv2D(filters, kernel_size, strides, padding='valid'))
    res.add(layers.BatchNormalization())
    return res


def CBA_block(filters: int, kernel_size: int = 3, strides: int = 1, padding: object = 'same'):
    res = Sequential()
    res.add(CB_block(filters, kernel_size, strides, padding))
    res.add(layers.ReLU())
    return res

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
            res.add(layers.Conv2D(16, 3, 1, 'same', activation='relu'))
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
        res.add(layers.Dense(self.class_number, activation='softmax'))
        return res
