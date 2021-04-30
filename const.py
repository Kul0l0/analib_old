#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : Kul0l0
@File    : const.py
'''
from tensorflow.keras import layers


DATA_FILE_TYPE = {'csv'}
# columns name
CN_FEATURE_NAME = 'Feature_name'
CN_TYPE = 'Data_type'
CN_EXPLANATION = 'Explanation'
CN_COUNT = 'Count'
CN_PERCENT = 'Percent'
CN_VALUE = 'Value'
CN_NUNIQUE = 'Nunique'
CN_MISSING_RATE = 'Missing_rate'
CN_FEATURE_LABEL = 'Feature_label'
# distribution
DB_INT_DIVIDE = 10
DB_OBJECT_DIVIDE = 20
# cut_bin
BIN_NUMBER = 10
# message
MSG_MORE_VALUE = 'more value'
MSG_UNTREATED = 'untreated'
MSG_USELESS = 'useless'
# output
OP_DEFAULT_FILE_NAME = 'default'
OP_EDA_EXTRA = '_EDA'

# template
EfficientNetB0 = [
#     ('TF',   layers.experimental.preprocessing.Rescaling(scale=1./255)),
    ('TF',   layers.experimental.preprocessing.Normalization()),
    # stem
    ('TF',   layers.ZeroPadding2D(padding=((0, 1), (0, 1)))),
    ('CBA',  {'filters':32, 'kernel_size':3, 'strides':2, 'padding':'valid', 'use_bias':False, 'activation': 'swish'}),
    # block1a
    ('CBA',  {'conv_type':'depthwise', 'kernel_size':3, 'strides':1, 'padding':'same', 'use_bias':False, 'activation': 'swish'}),
    ('SECB', {'r': 8, 'activation': 'swish', 'filters':16, 'kernel_size':1, 'strides':1, 'padding':'same', 'use_bias':False}),
    # block2a
    ('CBA',  {'filters':96, 'kernel_size':1, 'strides':1, 'padding':'same', 'use_bias':False, 'activation': 'swish'}),
    ('TF',   layers.ZeroPadding2D(padding=((0, 1), (0, 1)))),
    ('CBA',  {'conv_type':'depthwise', 'kernel_size':3, 'strides':2, 'padding':'valid', 'use_bias':False, 'activation': 'swish'}),
    ('SECB', {'r': 4, 'activation': 'swish', 'filters':24, 'kernel_size':1, 'strides':1, 'padding':'same', 'use_bias':False}),
    # block2b
    ('MBConv6', {'r': 6, 'activation': 'swish', 'filters':24, 'kernel_size':3, 'use_bias':False, 'rate':0.025}),
    # block3a
    ('CBA',  {'filters':144, 'kernel_size':1, 'strides':1, 'padding':'same', 'use_bias':False, 'activation': 'swish'}),
    ('TF',   layers.ZeroPadding2D(padding=((2, 2), (2, 2)))),
    ('CBA',  {'conv_type':'depthwise', 'kernel_size':5, 'strides':2, 'padding':'valid', 'use_bias':False, 'activation': 'swish'}),
    ('SECB', {'r': 6, 'activation': 'swish', 'filters':40, 'kernel_size':1, 'strides':1, 'padding':'same', 'use_bias':False}),
    # block3b
    ('MBConv6', {'r': 10, 'activation': 'swish', 'filters':40, 'kernel_size':5, 'use_bias':False, 'rate':0.05}),
    # block4a
    ('CBA',  {'filters':240, 'kernel_size':1, 'strides':1, 'padding':'same', 'use_bias':False, 'activation': 'swish'}),
    ('TF',   layers.ZeroPadding2D(padding=((1, 1), (1, 1)))),
    ('CBA',  {'conv_type':'depthwise', 'kernel_size':3, 'strides':2, 'padding':'valid', 'use_bias':False, 'activation': 'swish'}),
    ('SECB', {'r': 10, 'activation': 'swish', 'filters':80, 'kernel_size':1, 'strides':1, 'padding':'same', 'use_bias':False}),
    # block4b
    ('MBConv6', {'r': 20, 'activation': 'swish', 'filters':80, 'kernel_size':3, 'use_bias':False, 'rate':0.075}),
    # block4c
    ('MBConv6', {'r': 20, 'activation': 'swish', 'filters':80, 'kernel_size':3, 'use_bias':False, 'rate':0.0875}),
    # block5a
    ('CBA',  {'filters':480, 'kernel_size':1, 'strides':1, 'padding':'same', 'use_bias':False, 'activation': 'swish'}),
    ('CBA',  {'conv_type':'depthwise', 'kernel_size':5, 'strides':1, 'padding':'same', 'use_bias':False, 'activation': 'swish'}),
    ('SECB', {'r': 20, 'activation': 'swish', 'filters':112, 'kernel_size':1, 'strides':1, 'padding':'same', 'use_bias':False}),
    # block5b
    ('MBConv6', {'r': 28, 'activation': 'swish', 'filters':112, 'kernel_size':5, 'use_bias':False, 'rate':0.1125}),
    # block5c
    ('MBConv6', {'r': 28, 'activation': 'swish', 'filters':112, 'kernel_size':5, 'use_bias':False, 'rate':0.125}),
    # block6a
    ('CBA',  {'filters':672, 'kernel_size':1, 'strides':1, 'padding':'same', 'use_bias':False, 'activation': 'swish'}),
    ('TF',   layers.ZeroPadding2D(padding=((2, 2), (2, 2)))),
    ('CBA',  {'conv_type':'depthwise', 'kernel_size':5, 'strides':2, 'padding':'valid', 'use_bias':False, 'activation': 'swish'}),
    ('SECB', {'r': 28, 'activation': 'swish', 'filters':192, 'kernel_size':1, 'strides':1, 'padding':'same', 'use_bias':False}),
    # block6b
    ('MBConv6', {'r': 48, 'activation': 'swish', 'filters':192, 'kernel_size':5, 'use_bias':False, 'rate':0.15}),
    # block6c
    ('MBConv6', {'r': 48, 'activation': 'swish', 'filters':192, 'kernel_size':5, 'use_bias':False, 'rate':0.1625}),
    # block6d
    ('MBConv6', {'r': 48, 'activation': 'swish', 'filters':192, 'kernel_size':5, 'use_bias':False, 'rate':0.175}),
    # block7a
    ('CBA',  {'filters':1152, 'kernel_size':1, 'strides':1, 'padding':'same', 'use_bias':False, 'activation': 'swish'}),
    ('CBA',  {'conv_type':'depthwise', 'kernel_size':3, 'strides':1, 'padding':'same', 'use_bias':False, 'activation': 'swish'}),
    ('SECB', {'r': 48, 'activation': 'swish', 'filters':320, 'kernel_size':1, 'strides':1, 'padding':'same', 'use_bias':False}),
    # top
    ('CBA',  {'filters':1280, 'kernel_size':1, 'strides':1, 'padding':'same', 'use_bias':False, 'activation': 'swish'}),
    ('TF',   layers.GlobalAveragePooling2D()),
    ('TF',   layers.Dropout(rate=0.2)),
    ('TF',   layers.Dense(units=11, activation='sigmoid')),
]

