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

def CB_block(input, conv_type = 'normal', **kwargs):
    conv_func = {
        'normal': layers.Conv2D,
        'depthwise': layers.DepthwiseConv2D,
    }
    kernel_size, strides, padding = kwargs['kernel_size'], kwargs['strides'], kwargs['padding']
    output = (
        conv_func[conv_type](
            **kwargs,
            kernel_regularizer=keras.regularizers.L2(l2=0.0001),
            bias_regularizer=keras.regularizers.L2(l2=0.0001),
            name='K%d_S%d_P%s_%d'%(kernel_size, strides, padding, np.random.randint(5000)),
        )
    )(input)
    output = layers.BatchNormalization()(output)
    return output


def CBA_block(input, conv_type = 'normal', active_type = 'relu', **kwargs):
    output = CB_block(input, conv_type, **kwargs)
    output = layers.Activation(active_type)(output)
    return output


def pool_block(input, type: str = 'max', **kwargs):
    func_map = {
        'max': layers.MaxPool2D,
        'avg': layers.GlobalAveragePooling2D,
    }
    return func_map[type](**kwargs)(input)


def Res_plain_block(input, **kwargs):
    output = CBA_block(input, **kwargs)
    kwargs['strides'] = 1
    output = CBA_block(output, **kwargs)
    return output


def Res_common_block(input, type: str=None, **kwargs):
    filters, kernel_size, strides, padding = kwargs['filters'], kwargs['kernel_size'], kwargs['strides'], kwargs['padding']
    shortcut = CB_block(input, filters=filters, kernel_size=1, strides=strides, padding='valid') if type=='edge' else input
    output = CBA_block(input, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)
    output = CB_block(output, filters=filters, kernel_size=kernel_size, strides=1, padding=padding)
    return layers.ReLU()(layers.add([shortcut, output]))


def Res_bottleneck_block(input, type: str=None, **kwargs):
    filters, kernel_size, strides, padding = kwargs['filters'], kwargs['kernel_size'], kwargs['strides'], kwargs['padding']
    output = CBA_block(input, filters=filters, kernel_size=1, strides=strides, padding='valid')
    output = CBA_block(output, filters=filters, kernel_size=kernel_size, strides=1, padding=padding)
    output = CB_block(output, filters=filters*4, kernel_size=1, strides=1, padding='valid')
    shortcut = CB_block(input, filters=filters*4, kernel_size=1, strides=strides, padding='valid') if type=='edge' else input
    return layers.ReLU()(layers.add([shortcut, output]))


def top_block(input, class_number, activation = 'softmax'):
    output = layers.GlobalAveragePooling2D()(input)
    output = layers.Dense(
        units=class_number,
        activation=activation,
        kernel_regularizer=keras.regularizers.L2(l2=0.0001)
    )(output)
    return  output

def SE_block(input, ratio):
    channel = input.shape[-1]
    output = layers.GlobalAveragePooling2D()(input)
    output = layers.Dense(int(channel*ratio), activation='relu')(output)
    output = layers.Dense(channel, activation='sigmoid')(output)
    mul = layers.Multiply()
    return mul([input, output])

# def Res_common_SE_block(input, filters: int, kernel_size: int = 3, strides: int = 1, padding: object = 'same', type: str=None, ratio=0.5):
#     shortcut = CB_block(input, filters, kernel_size=1, strides=strides, padding='valid') if type=='edge' else input
#     output = CBA_block(input, filters, kernel_size, strides, padding=padding)
#     output = CB_block(output, filters, kernel_size, strides=1, padding=padding)
#     output = SE_block(output, ratio)
#     return layers.ReLU()(layers.add([shortcut, output]))

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
    #'Res_common_SE':    Res_common_SE_block,
}

def build(input_shape, config_list: list):
    input, output = keras.Input(shape=input_shape), None
    for block, config in config_list:
        if isinstance(block, str):
            times = 1
        else:
            block, times = block
        for i in range(times):
            if block != 'TF':
                config['input'] = input if output is None else output
                output = BLOCK_MAP[block](**config)
            else:
                # if block is "TF", config will be tf layers object
                output = input if output is None else output
                output = config(output)
    return keras.Model(inputs=input, outputs=output)

def block_list():
    print(list(BLOCK_MAP.keys()))
