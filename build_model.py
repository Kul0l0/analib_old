#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : Kul0l0
@File    : cnn.py
@Time    : 2021/3/8 下午3:16
'''
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import const


def CB_block(input,  filters: int, kernel_size: int, strides: int = 1, padding: object = 'valid', conv_type = 'normal'):
    conv_func = {
        'normal': layers.Conv2D,
        'depthwise': layers.DepthwiseConv2D
    }
    if padding in ('same', 'valid'):
        output = (
            conv_func[conv_type](
                filters, kernel_size, strides, padding=padding,
                kernel_regularizer=keras.regularizers.L2(l2=0.0001),
                bias_regularizer=keras.regularizers.L2(l2=0.0001),
                name='F%d_K%d_S%d_P%s_%0.4d'%(filters, kernel_size, strides, padding, np.random.randint(5000)),
            )
        )(input)
    else:
        output = layers.ZeroPadding2D(padding=padding)(input)
        output = (
            conv_func[conv_type](
                filters, kernel_size, strides, padding='valid',
                kernel_regularizer=keras.regularizers.L2(l2=0.0001),
                bias_regularizer=keras.regularizers.L2(l2=0.0001)
            )
        )(output)
    output = layers.BatchNormalization()(output)
    return output


def CBA_block(input, filters: int, kernel_size: int, strides: int = 1, padding: object = 'valid', conv_type = 'normal'):
    output = CB_block(input, filters, kernel_size, strides, padding, conv_type)
    output = layers.ReLU()(output)
    return output


def pool_block(input, type: str = 'max', pool_size: int = 2, strides: int = 1, padding: object = 'valid'):
    func_map = {
        'max': layers.MaxPool2D,
        'avg': layers.GlobalAveragePooling2D,
    }
    return func_map[type](pool_size, strides, padding)(input)


def Res_plain_block(input, filters: int, kernel_size: int = 3, strides: int = 1, padding: object = 'same'):
    output = CBA_block(input, filters, kernel_size, strides, padding=padding)
    output = CBA_block(output, filters, kernel_size, strides=1, padding=padding)
    return output


def Res_common_block(input, filters: int, kernel_size: int = 3, strides: int = 1, padding: object = 'same', type: str=None):
    shortcut = CB_block(input, filters, kernel_size=1, strides=strides, padding='valid') if type=='edge' else input
    output = CBA_block(input, filters, kernel_size, strides, padding=padding)
    output = CB_block(output, filters, kernel_size, strides=1, padding=padding)
    return layers.ReLU()(layers.add([shortcut, output]))


def Res_bottleneck_block(input, filters: int, kernel_size: int = 3, strides: int = 1, padding: object = 'same', type: str=None):
    shortcut = CB_block(input, filters*4, kernel_size=1, strides=strides, padding='valid') if type=='edge' else input
    output = CBA_block(input, filters, kernel_size=1, strides=strides, padding='valid')
    output = CBA_block(output, filters, kernel_size, strides=1, padding=padding)
    output = CB_block(output, filters*4, kernel_size=1, strides=1, padding='valid')
    return layers.ReLU()(layers.add([shortcut, output]))


def top_block(input, class_number, activation = 'softmax'):
    output = layers.GlobalAveragePooling2D()(input)
    output = layers.Dense(units=class_number, activation=activation, kernel_regularizer=keras.regularizers.L2(l2=0.0001))(output)
    return  output

def SE_block(input, ratio):
    channel = input.shape[-1]
    output = layers.GlobalAveragePooling2D()(input)
    output = layers.Dense(int(channel*ratio), activation='relu')(output)
    output = layers.Dense(channel, activation='sigmoid')(output)
    mul = layers.Multiply()
    return mul([input, output])

def Res_common_SE_block(input, filters: int, kernel_size: int = 3, strides: int = 1, padding: object = 'same', type: str=None, ratio=0.5):
    shortcut = CB_block(input, filters, kernel_size=1, strides=strides, padding='valid') if type=='edge' else input
    output = CBA_block(input, filters, kernel_size, strides, padding=padding)
    output = CB_block(output, filters, kernel_size, strides=1, padding=padding)
    output = SE_block(output, ratio)
    return layers.ReLU()(layers.add([shortcut, output]))

def MBConv6_block(input, filters: int, kernel_size: int = 3, strides: int = 1, padding: object = 'same', type: str=None):
    output = CBA_block(input, filters, kernel_size=1, strides=strides, padding=padding, conv_type='normal')
    output = CBA_block(output, filters, kernel_size, strides, padding=padding, conv_type='depthwise')
    output = CB_block(output, filters, kernel_size=1, strides=1, padding=padding, conv_type='normal')
    return layers.ReLU()(layers.add([input, output]))

# block_dict
BLOCK_MAP = {
    'CB':               CB_block,
    'CBA':              CBA_block,
    'Res_plain':        Res_plain_block,
    'Res_common':       Res_common_block,
    'Res_bottleneck':   Res_bottleneck_block,
    'top':              top_block,
    'SE':               SE_block,
    'Res_common_SE':    Res_common_SE_block,
}

def build(input_shape, config_list: list):
    input, output = keras.Input(shape=input_shape), None
    for block, config in config_list:
        if isinstance(block, str):
            times = 1
        else:
            block, times = block
        for i in range(times):
            config['input'] = input if output is None else output
            output = BLOCK_MAP[block](**config)
    return keras.Model(inputs=input, outputs=output)

def block_list():
    print(list(BLOCK_MAP.keys()))
