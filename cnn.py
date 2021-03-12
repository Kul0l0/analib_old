#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : Kul0l0
@File    : cnn.py
@Time    : 2021/3/8 下午3:16
'''
from tensorflow.keras import Model
from tensorflow.keras import layers


class Resnet(Model):
    def __init__(self, data_set, block_type, img_size, class_number, stack_list, filter_list):
        super(Resnet, self).__init__()
        self.data_set = data_set
        self.block_type = block_type
        self.img_size = img_size
        self.class_number = class_number
        self.stack_list = stack_list
        self.filter_list = filter_list
        #self.install()

    def call(self, inputs, training=None, mask=None):
        outputs = self.install_head(inputs)
        outputs = self.install_body(outputs)
        return self.install_tail(outputs)

    def CB_block(self, inputs, filters: int, kernel_size: int=3, strides:int=1, padding:object='valid'):
        if padding in ('same', 'valid'):
            outputs = layers.Conv2D(filters, kernel_size, strides, padding=padding)(inputs)
        else:
            outputs = layers.ZeroPadding2D(padding=padding)(inputs)
            outputs = layers.Conv2D(filters, kernel_size, strides, padding='valid')(outputs)
        return layers.BatchNormalization()(outputs)

    def CBA_block(self, inputs, filters:int, kernel_size:int=3, strides:int=1, padding:object='valid'):
        outputs = self.CB_block(inputs, filters, kernel_size, strides, padding)
        return layers.ReLU()(outputs)

    def plain_block(self, inputs, begin, filters: int, kernel_size:int=3, strides:int=1):
        outputs = self.CBA_block(inputs, filters, kernel_size, strides=2, padding='same')
        outputs = self.CBA_block(outputs, filters, kernel_size, strides, padding='same')
        return outputs

    def common_block(self, inputs, begin, filters:int, kernel_size:int=3, strides:int=1):
        pass

    def bottleneck_block(self, inputs, begin, filters:int, kernel_size:int=3, strides:int=1):
        pass

    def install_head(self, inputs):
        #inputs = layers.Input(shape=(self.img_size, self.img_size, 3))
        if self.data_set == 'cifar10':
            outputs = layers.Conv2D(16, 3, 2, 'same', activation='relu')(inputs)
        else: # imagenet
            outputs = self.CBA_block(inputs, 64, 7, 2, padding=7)(inputs)
            outputs = layers.MaxPool2D(pool_size=3, strides=2, padding=((1,1),(1,1)))(outputs)
        return outputs

    def install_body(self, inputs):
        block_map = {
            'plain': self.plain_block,
            'common': self.common_block,
            'bottleneck': self.bottleneck_block,
        }
        block = block_map[self.block_type]
        outputs = inputs
        for idx, (filters, stack) in enumerate(zip(self.filter_list, self.stack_list)):
            for i in range(stack):
                if i == 0:
                    if idx == 0:
                        outputs = block(inputs=outputs, begin=True, filters=filters, kernel_size=3, strides=2)
                    else:
                        outputs = block(inputs=outputs, begin=False, filters=filters, kernel_size=3, strides=2)
                else:
                    outputs = block(inputs=outputs, begin=False, filters=filters, kernel_size=3, strides=1)
        return outputs

    def install_tail(self, inputs):
        outputs = layers.GlobalAveragePooling2D()(inputs)
        return layers.Dense(self.class_number)(outputs)

    def install(self):
        head_inputs, head_outputs = self.install_head()
        body_outputs = self.install_body(head_outputs)
        tail_outputs = self.install_tail(body_outputs)
        return Model(head_inputs, tail_outputs)





