#!/usr/bin/env python
# encoding: utf-8
"""
@Author  : Kul0l0
@Time    : 2021/3/8 下午3:16
"""
from tensorflow import keras
from tensorflow.keras import layers, Sequential
import random
import string


def random_str(length) -> str:
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(length))


def conv2d_bn(**kwargs) -> Sequential:
    k, s = kwargs['kernel_size'], kwargs['strides']
    return Sequential(
        layers=[
            layers.Conv2D(**kwargs),
            layers.BatchNormalization(),
        ],
        name='Conv2D_k%d_s%d_BN_%s' % (k, s, random_str(3))
    )


def conv2d_bn_dp(**kwargs):
    k, s = kwargs['kernel_size'], kwargs['strides']
    rate = kwargs.pop('rate')
    name = 'Conv2D_k%d_s%d_BN_D_%s_%s' % (k, s, str(rate), random_str(3))
    # make sequential
    seq = Sequential(name=name)
    seq.add(conv2d_bn(**kwargs))
    seq.add(layers.Dropout(rate=rate))
    return seq


def conv2d_bn_a(**kwargs) -> Sequential:
    k, s = kwargs['kernel_size'], kwargs['strides']
    activation = kwargs.pop('activation')
    name = 'Conv2D_k%d_s%d_BN_A_%s_%s' % (k, s, activation, random_str(3))
    # make sequential
    seq = Sequential(name=name)
    seq.add(conv2d_bn(**kwargs))
    seq.add(layers.Activation(activation=activation))
    return seq


def pool_block(block_input, style: str = 'max', **kwargs):
    pool_map = {
        'max': layers.MaxPool2D,
        'avg': layers.GlobalAveragePooling2D,
    }
    return pool_map[style](**kwargs)(block_input)


def res_plain_block(**kwargs):
    k, s = kwargs['kernel_size'], kwargs['strides']
    name = 'Plain_k%d_s%d_%s' % (k, s, random_str(3))
    seq = Sequential(name=name)
    seq.add(conv2d_bn_a(**kwargs))
    kwargs['strides'] = 1
    seq.add(conv2d_bn_a(**kwargs))
    return seq


def res_common_block(block_input, upsample: bool = False, **kwargs):
    # trunk
    block_output = conv2d_bn_a(block_input, **kwargs)
    kwargs['strides'] = 1
    block_output = conv2d_bn(block_output, **kwargs)
    # shortcut
    shortcut = block_input
    if upsample:
        kwargs['strides'] = 2
        kwargs['kernel_size'] = 1
        kwargs['padding'] = 'valid'
        shortcut = conv2d_bn(block_input, **kwargs)
    return layers.ReLU()(layers.add([shortcut, block_output]))


def res_bottleneck_block(block_input, upsample: bool = False, **kwargs):
    # trunk
    filters, kernel_size, strides = kwargs['filters'], kwargs['kernel_size'], kwargs['strides']
    kwargs['filters'], kwargs['kernel_size'], kwargs['strides'], kwargs['padding'] = filters // 4, 1, 1, 'valid'
    block_output = conv2d_bn_a(block_input, **kwargs)
    kwargs['filters'], kwargs['kernel_size'], kwargs['strides'], kwargs[
        'padding'] = filters // 4, kernel_size, strides, 'same'
    block_output = conv2d_bn_a(block_output, **kwargs)
    kwargs['filters'], kwargs['kernel_size'], kwargs['strides'], kwargs['padding'] = filters, 1, 1, 'valid'
    block_output = conv2d_bn(block_output, **kwargs)
    # shortcut
    shortcut = block_input
    if upsample:
        kwargs['strides'] = 2
        shortcut = conv2d_bn(block_input, **kwargs)
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
    block_output = conv2d_bn(block_input, activation=activation, filters=filters // 2, kernel_size=1, strides=1,
                             padding=padding, use_bias=use_bias)
    block_output = conv2d_bn_a(block_output, activation=activation, filters=filters, kernel_size=kernel_size,
                               strides=strides, padding=padding, use_bias=use_bias)
    return layers.ReLU()(layers.add([shortcut, block_output]))


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
    block_output = conv2d_bn(block_output, conv_type, **kwargs)
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
    block_output = conv2d_bn_a(block_input, activation=activation, filters=filters * 6, kernel_size=1, strides=1,
                               padding='same', use_bias=use_bias)
    block_output = conv2d_bn_a(block_output, activation=activation, conv_type='depthwise', kernel_size=kernel_size,
                               strides=1, padding='same', use_bias=use_bias)
    block_output = se_block(block_output, r, activation=activation)
    block_output = conv2d_bn_dp(block_output, rate=rate, filters=filters, kernel_size=1, strides=1, padding='same',
                             use_bias=use_bias)
    return layers.add([block_input, block_output])


def top_block(block_input, class_number, activation='softmax'):
    block_output = layers.GlobalAveragePooling2D()(block_input)
    block_output = layers.Dense(
        units=class_number,
        activation=activation,
        kernel_regularizer=keras.regularizers.L2(l2=0.0001)
    )(block_output)
    return block_output


# block_dict
BLOCK_MAP = {
    'CB': conv2d_bn,
    'CBA': conv2d_bn_a,
    'CBD': conv2d_bn_dp,
    'res_plain': res_plain_block,
    'res_common': res_common_block,
    'res_bottleneck': res_bottleneck_block,
    'dark_res': dark_res_block,
    'pool': pool_block,
    'top': top_block,
    'SE': se_block,
    'SECB': secb_block,
    'MBConv6': mbconv6_block,
    # 'res_common_SE':    res_common_se_block,
}


def block_list():
    print(list(BLOCK_MAP.keys()))
