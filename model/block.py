#!/usr/bin/env python
# encoding: utf-8
"""
@Author  : Kul0l0
@Time    : 2021/3/8 下午3:16
"""
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

"""
###memo###
c: Conv2D
b: BatchNormalization
a: Activation
"""


def cb_block(block_input, conv_type='normal', **kwargs):
    """
    the most basic block, Conv2D and BatchNormalization
    :param block_input: block input
    :type block_input: image data
    :param conv_type: conv type, the key of conv_func
    :type conv_type: str
    :param kwargs: param of conv2D
    :type kwargs: **
    :return:
    :rtype:
    """
    conv_func = {
        'normal': layers.Conv2D,
        'depthwise': layers.DepthwiseConv2D,
    }
    kernel_size, strides, padding = kwargs['kernel_size'], kwargs['strides'], kwargs['padding']
    block_output = (
        conv_func[conv_type](
            **kwargs,
            # kernel_regularizer=keras.regularizers.L2(l2=0.0001),
            # bias_regularizer=keras.regularizers.L2(l2=0.0001),
            name='K%d_S%d_P%s_%d' % (kernel_size, strides, padding, np.random.randint(50000)),
        )
    )(block_input)
    return layers.BatchNormalization()(block_output)


def cbd_block(block_input, conv_type='normal', rate=1.0, **kwargs):
    # TODO: add a code comment
    block_output = cb_block(block_input, conv_type, **kwargs)
    block_output = layers.Dropout(rate=rate)(block_output)
    return block_output


def cba_block(block_input, conv_type='normal', activation='relu', **kwargs):
    block_output = cb_block(block_input, conv_type, **kwargs)
    block_output = layers.Activation(activation)(block_output)
    return block_output


def pool_block(block_input, pool_type: str = 'max', **kwargs):
    func_map = {
        'max': layers.MaxPool2D,
        'avg': layers.GlobalAveragePooling2D,
    }
    return func_map[pool_type](**kwargs)(block_input)


def res_plain_block(block_input, **kwargs):
    block_output = cba_block(block_input, **kwargs)
    kwargs['strides'] = 1
    block_output = cba_block(block_output, **kwargs)
    return block_output


def res_common_block(block_input, block_location: str = None, **kwargs):
    activation, filters, kernel_size, strides, padding = kwargs['activation'], kwargs['filters'], kwargs['kernel_size'], kwargs['strides'], kwargs['padding']
    shortcut = cb_block(block_input, activation=activation, filters=filters, kernel_size=1, strides=strides, padding='valid') if block_location == 'edge' else block_input
    block_output = cba_block(block_input, activation=activation, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)
    block_output = cb_block(block_output, activation=activation, filters=filters, kernel_size=kernel_size, strides=1, padding=padding)
    return layers.ReLU()(layers.add([shortcut, block_output]))


def res_bottleneck_block(block_input, block_location: str = None, **kwargs):
    activation, filters, kernel_size, strides, padding = kwargs['activation'], kwargs['filters'], kwargs['kernel_size'], kwargs['strides'], kwargs['padding']
    block_output = cba_block(block_input, activation=activation, filters=filters, kernel_size=1, strides=strides, padding='valid')
    block_output = cba_block(block_output, activation=activation, filters=filters, kernel_size=kernel_size, strides=1, padding=padding)
    block_output = cb_block(block_output, activation=activation, filters=filters * 4, kernel_size=1, strides=1, padding='valid')
    shortcut = cb_block(block_input, activation=activation, filters=filters * 4, kernel_size=1, strides=strides, padding='valid') if block_location == 'edge' else block_input
    return layers.ReLU()(layers.add([shortcut, block_output]))


def dark_res_block(block_input, **kwargs):
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
    shortcut = block_input
    block_output = cb_block(block_input, activation=activation, filters=filters // 2, kernel_size=1, strides=1, padding=padding, use_bias=use_bias)
    block_output = cba_block(block_output, activation=activation, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias)
    return layers.ReLU()(layers.add([shortcut, block_output]))


def top_block(block_input, class_number, activation='softmax'):
    block_output = layers.GlobalAveragePooling2D()(block_input)
    block_output = layers.Dense(
        units=class_number,
        activation=activation,
        kernel_regularizer=keras.regularizers.L2(l2=0.0001)
    )(block_output)
    return block_output


def se_block(block_input, r, activation='relu'):
    channel = block_input.shape[-1]
    reduce_filter = int(channel * r) if isinstance(r, float) else r
    block_output = layers.GlobalAveragePooling2D()(block_input)
    block_output = layers.Reshape((1, 1, channel))(block_output)
    block_output = layers.Conv2D(filters=reduce_filter, kernel_size=1, activation=activation)(block_output)
    block_output = layers.Conv2D(filters=channel, kernel_size=1, activation='sigmoid')(block_output)
    mul = layers.Multiply()
    return mul([block_input, block_output])


def secb_block(block_input, r, conv_type='normal', activation='relu', **kwargs):
    block_output = se_block(block_input, r=r, activation=activation)
    block_output = cb_block(block_output, conv_type, **kwargs)
    return block_output


def mbconv6_block(block_input, **kwargs):
    activation, filters, kernel_size, r, rate, use_bias = (
        kwargs['activation'],
        kwargs['filters'],
        kwargs['kernel_size'],
        kwargs['r'],
        kwargs['rate'],
        kwargs['use_bias'],
    )
    block_output = cba_block(block_input, activation=activation, filters=filters * 6, kernel_size=1, strides=1, padding='same', use_bias=use_bias)
    block_output = cba_block(block_output, activation=activation, conv_type='depthwise', kernel_size=kernel_size, strides=1, padding='same', use_bias=use_bias)
    block_output = se_block(block_output, r, activation=activation)
    block_output = cbd_block(block_output, rate=rate, filters=filters, kernel_size=1, strides=1, padding='same', use_bias=use_bias)
    return layers.add([block_input, block_output])


# block_dict
BLOCK_MAP = {
    'CB': cb_block,
    'CBA': cba_block,
    'CBD': cbd_block,
    'res_plain': res_plain_block,
    'res_common': res_common_block,
    'res_bottleneck': res_bottleneck_block,
    'dark_res': dark_res_block,
    'top': top_block,
    'SE': se_block,
    'SECB': secb_block,
    'MBConv6': mbconv6_block,
    # 'res_common_SE':    res_common_se_block,
}


def build(block_input, config_list: list, return_model: bool = True, name=None):
    if isinstance(block_input, int) or block_input is None:
        block_input = keras.Input(shape=(block_input, block_input, 3))
    block_output = None
    for block, config in config_list:
        if isinstance(block, str):
            times = 1
        else:
            block, times = block
        for i in range(times):
            # not TF layers
            if block != 'TF':
                config['block_input'] = block_input if block_output is None else block_output
                block_output = BLOCK_MAP[block](**config)
            # if block is "TF", config will be tf layers object
            else:
                block_output = block_input if block_output is None else block_output
                block_output = config(block_output)
    if return_model:
        return keras.Model(inputs=block_input, block_outputs=block_output, name=name)
    else:
        return block_input, block_output


def block_list():
    print(list(BLOCK_MAP.keys()))
