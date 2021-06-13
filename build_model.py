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
            #kernel_regularizer=keras.regularizers.L2(l2=0.0001),
            #bias_regularizer=keras.regularizers.L2(l2=0.0001),
            name='K%d_S%d_P%s_%d'%(kernel_size, strides, padding, np.random.randint(50000)),
        )
    )(input)
    output = layers.BatchNormalization()(output)
    return output


def CBD_block(input, conv_type = 'normal', rate=1.0, **kwargs):
    output = CB_block(input, conv_type, **kwargs)
    output = layers.Dropout(rate=rate)(output)
    return output


def CBA_block(input, conv_type = 'normal', activation = 'relu', **kwargs):
    output = CB_block(input, conv_type, **kwargs)
    output = layers.Activation(activation)(output)
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
    activation, filters, kernel_size, strides, padding = kwargs['activation'], kwargs['filters'], kwargs['kernel_size'], kwargs['strides'], kwargs['padding']
    shortcut = CB_block(input, activation=activation, filters=filters, kernel_size=1, strides=strides, padding='valid') if type=='edge' else input
    output = CBA_block(input, activation=activation, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)
    output = CB_block(output, activation=activation, filters=filters, kernel_size=kernel_size, strides=1, padding=padding)
    return layers.ReLU()(layers.add([shortcut, output]))


def Res_bottleneck_block(input, type: str=None, **kwargs):
    activation, filters, kernel_size, strides, padding = kwargs['activation'], kwargs['filters'], kwargs['kernel_size'], kwargs['strides'], kwargs['padding']
    output = CBA_block(input, activation=activation, filters=filters, kernel_size=1, strides=strides, padding='valid')
    output = CBA_block(output, activation=activation, filters=filters, kernel_size=kernel_size, strides=1, padding=padding)
    output = CB_block(output, activation=activation, filters=filters*4, kernel_size=1, strides=1, padding='valid')
    shortcut = CB_block(input, activation=activation, filters=filters*4, kernel_size=1, strides=strides, padding='valid') if type=='edge' else input
    return layers.ReLU()(layers.add([shortcut, output]))

def Dark_res_block(input, type: str=None, **kwargs):
    (
        activation,
        filters,
        kernel_size,
        strides,
        padding,
        use_bias,
    ) = (
        kwargs['activation'],
        kwargs['filters'],
        kwargs['kernel_size'],
        kwargs['strides'],
        kwargs['padding'],
        kwargs['use_bias'],
    )
    shortcut = input
    output = CB_block(input, activation=activation, filters=filters//2, kernel_size=1, strides=1, padding=padding, use_bias=use_bias)
    output = CBA_block(output, activation=activation, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias)
    return layers.ReLU()(layers.add([shortcut, output]))

def top_block(input, class_number, activation = 'softmax'):
    output = layers.GlobalAveragePooling2D()(input)
    output = layers.Dense(
        units=class_number,
        activation=activation,
        kernel_regularizer=keras.regularizers.L2(l2=0.0001)
    )(output)
    return  output

def SE_block(input, r, activation='relu'):
    channel = input.shape[-1]
    reduce_filter = int(channel*r)  if isinstance(r, float) else r
    output = layers.GlobalAveragePooling2D()(input)
    output = layers.Reshape((1,1,channel))(output)
    output = layers.Conv2D(filters=reduce_filter, kernel_size=1, activation=activation)(output)
    output = layers.Conv2D(filters=channel, kernel_size=1, activation='sigmoid')(output)
    mul = layers.Multiply()
    return mul([input, output])


def SECB_block(input, r, conv_type = 'normal', activation = 'relu', **kwargs):
    output = SE_block(input, r=r, activation=activation)
    output = CB_block(output, conv_type, **kwargs)
    return output


def MBConv6_block(input, **kwargs):
    activation, filters, kernel_size, r, rate, use_bias = (
        kwargs['activation'],
        kwargs['filters'],
        kwargs['kernel_size'],
        kwargs['r'],
        kwargs['rate'],
        kwargs['use_bias'],
    )
    output = CBA_block(input, activation=activation, filters=filters*6, kernel_size=1, strides=1, padding='same', use_bias=use_bias)
    output = CBA_block(output, activation=activation, conv_type='depthwise', kernel_size=kernel_size, strides=1, padding='same', use_bias=use_bias)
    output = SE_block(output, r, activation=activation)
    output = CBD_block(output, rate=rate, filters=filters, kernel_size=1, strides=1, padding='same', use_bias=use_bias)
    return layers.add([input, output])


# block_dict
BLOCK_MAP = {
    'CB':               CB_block,
    'CBA':              CBA_block,
    'CBD':              CBD_block,
    'Res_plain':        Res_plain_block,
    'Res_common':       Res_common_block,
    'Res_bottleneck':   Res_bottleneck_block,
    'Dark_res':         Dark_res_block,
    'top':              top_block,
    'SE':               SE_block,
    'SECB':             SECB_block,
    'MBConv6':          MBConv6_block,
    #'Res_common_SE':    Res_common_SE_block,
}

def build(input, config_list: list, return_model: bool=True, name=None):
    if isinstance(input, int) or input is None:
        input = keras.Input(shape=(input, input, 3))
    output = None
    for block, config in config_list:
        if isinstance(block, str):
            times = 1
        else:
            block, times = block
        for i in range(times):
            # not TF layers
            if block != 'TF':
                config['input'] = input if output is None else output
                output = BLOCK_MAP[block](**config)
            # if block is "TF", config will be tf layers object
            else:
                output = input if output is None else output
                output = config(output)
    if return_model:
        return keras.Model(inputs=input, outputs=output, name=name)
    else:
        return input, output

def block_list():
    print(list(BLOCK_MAP.keys()))
