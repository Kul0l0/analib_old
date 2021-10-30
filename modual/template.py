#!/usr/bin/env python
# encoding: utf-8
"""
@Author  : Kul0l0
@File    : template.py
@Time    : 2021/6/7 上午9:16
"""
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
from experiment import block


# template
EfficientNetB0 = [
    #     ('TF',   layers.experimental.preprocessing.Rescaling(scale=1./255)),
    ('TF', layers.experimental.preprocessing.Normalization()),
    # stem
    ('TF', layers.ZeroPadding2D(padding=((0, 1), (0, 1)))),
    ('CBA', {'filters': 32, 'kernel_size': 3, 'strides': 2, 'padding': 'valid', 'use_bias': False, 'activation': 'swish'}),
    # block1a
    ('CBA', {'conv_type': 'depthwise', 'kernel_size': 3, 'strides': 1, 'padding': 'same', 'use_bias': False, 'activation': 'swish'}),
    ('SECB', {'r': 8, 'activation': 'swish', 'filters': 16, 'kernel_size': 1, 'strides': 1, 'padding': 'same', 'use_bias': False}),
    # block2a
    ('CBA', {'filters': 96, 'kernel_size': 1, 'strides': 1, 'padding': 'same', 'use_bias': False, 'activation': 'swish'}),
    ('TF', layers.ZeroPadding2D(padding=((0, 1), (0, 1)))),
    ('CBA', {'conv_type': 'depthwise', 'kernel_size': 3, 'strides': 2, 'padding': 'valid', 'use_bias': False, 'activation': 'swish'}),
    ('SECB', {'r': 4, 'activation': 'swish', 'filters': 24, 'kernel_size': 1, 'strides': 1, 'padding': 'same', 'use_bias': False}),
    # block2b
    ('MBConv6', {'r': 6, 'activation': 'swish', 'filters': 24, 'kernel_size': 3, 'use_bias': False, 'rate': 0.025}),
    # block3a
    ('CBA', {'filters': 144, 'kernel_size': 1, 'strides': 1, 'padding': 'same', 'use_bias': False, 'activation': 'swish'}),
    ('TF', layers.ZeroPadding2D(padding=((2, 2), (2, 2)))),
    ('CBA', {'conv_type': 'depthwise', 'kernel_size': 5, 'strides': 2, 'padding': 'valid', 'use_bias': False, 'activation': 'swish'}),
    ('SECB', {'r': 6, 'activation': 'swish', 'filters': 40, 'kernel_size': 1, 'strides': 1, 'padding': 'same', 'use_bias': False}),
    # block3b
    ('MBConv6', {'r': 10, 'activation': 'swish', 'filters': 40, 'kernel_size': 5, 'use_bias': False, 'rate': 0.05}),
    # block4a
    ('CBA', {'filters': 240, 'kernel_size': 1, 'strides': 1, 'padding': 'same', 'use_bias': False, 'activation': 'swish'}),
    ('TF', layers.ZeroPadding2D(padding=((1, 1), (1, 1)))),
    ('CBA', {'conv_type': 'depthwise', 'kernel_size': 3, 'strides': 2, 'padding': 'valid', 'use_bias': False, 'activation': 'swish'}),
    ('SECB', {'r': 10, 'activation': 'swish', 'filters': 80, 'kernel_size': 1, 'strides': 1, 'padding': 'same', 'use_bias': False}),
    # block4b
    ('MBConv6', {'r': 20, 'activation': 'swish', 'filters': 80, 'kernel_size': 3, 'use_bias': False, 'rate': 0.075}),
    # block4c
    ('MBConv6', {'r': 20, 'activation': 'swish', 'filters': 80, 'kernel_size': 3, 'use_bias': False, 'rate': 0.0875}),
    # block5a
    ('CBA', {'filters': 480, 'kernel_size': 1, 'strides': 1, 'padding': 'same', 'use_bias': False, 'activation': 'swish'}),
    ('CBA', {'conv_type': 'depthwise', 'kernel_size': 5, 'strides': 1, 'padding': 'same', 'use_bias': False, 'activation': 'swish'}),
    ('SECB', {'r': 20, 'activation': 'swish', 'filters': 112, 'kernel_size': 1, 'strides': 1, 'padding': 'same', 'use_bias': False}),
    # block5b
    ('MBConv6', {'r': 28, 'activation': 'swish', 'filters': 112, 'kernel_size': 5, 'use_bias': False, 'rate': 0.1125}),
    # block5c
    ('MBConv6', {'r': 28, 'activation': 'swish', 'filters': 112, 'kernel_size': 5, 'use_bias': False, 'rate': 0.125}),
    # block6a
    ('CBA', {'filters': 672, 'kernel_size': 1, 'strides': 1, 'padding': 'same', 'use_bias': False, 'activation': 'swish'}),
    ('TF', layers.ZeroPadding2D(padding=((2, 2), (2, 2)))),
    ('CBA', {'conv_type': 'depthwise', 'kernel_size': 5, 'strides': 2, 'padding': 'valid', 'use_bias': False, 'activation': 'swish'}),
    ('SECB', {'r': 28, 'activation': 'swish', 'filters': 192, 'kernel_size': 1, 'strides': 1, 'padding': 'same', 'use_bias': False}),
    # block6b
    ('MBConv6', {'r': 48, 'activation': 'swish', 'filters': 192, 'kernel_size': 5, 'use_bias': False, 'rate': 0.15}),
    # block6c
    ('MBConv6', {'r': 48, 'activation': 'swish', 'filters': 192, 'kernel_size': 5, 'use_bias': False, 'rate': 0.1625}),
    # block6d
    ('MBConv6', {'r': 48, 'activation': 'swish', 'filters': 192, 'kernel_size': 5, 'use_bias': False, 'rate': 0.175}),
    # block7a
    ('CBA', {'filters': 1152, 'kernel_size': 1, 'strides': 1, 'padding': 'same', 'use_bias': False, 'activation': 'swish'}),
    ('CBA', {'conv_type': 'depthwise', 'kernel_size': 3, 'strides': 1, 'padding': 'same', 'use_bias': False, 'activation': 'swish'}),
    ('SECB', {'r': 48, 'activation': 'swish', 'filters': 320, 'kernel_size': 1, 'strides': 1, 'padding': 'same', 'use_bias': False}),
    # top
    ('CBA', {'filters': 1280, 'kernel_size': 1, 'strides': 1, 'padding': 'same', 'use_bias': False, 'activation': 'swish'}),
    ('TF', layers.GlobalAveragePooling2D()),
    ('TF', layers.Dropout(rate=0.2)),
    ('TF', layers.Dense(units=11, activation='sigmoid')),
]


def Darknet53():
    structure_256 = [
        ('CBA', {'filters': 32, 'kernel_size': 3, 'strides': 1, 'padding': 'same', 'use_bias': False, 'activation': 'relu'}),
        ('CBA', {'filters': 64, 'kernel_size': 3, 'strides': 2, 'padding': 'same', 'use_bias': False, 'activation': 'relu'}),
        ('Dark_res', {'filters': 64, 'kernel_size': 3, 'strides': 1, 'padding': 'same', 'use_bias': False, 'activation': 'relu'}),
        ('CBA', {'filters': 128, 'kernel_size': 3, 'strides': 2, 'padding': 'same', 'use_bias': False, 'activation': 'relu'}),
        (('Dark_res', 2), {'filters': 128, 'kernel_size': 3, 'strides': 1, 'padding': 'same', 'use_bias': False, 'activation': 'relu'}),
        ('CBA', {'filters': 256, 'kernel_size': 3, 'strides': 2, 'padding': 'same', 'use_bias': False, 'activation': 'relu'}),
        (('Dark_res', 8), {'filters': 256, 'kernel_size': 3, 'strides': 1, 'padding': 'same', 'use_bias': False, 'activation': 'relu'}),
    ]
    input, output_256 = block.build(None, structure_256, return_model=False)
    structure_512 = [
        ('CBA',
         {'filters': 512, 'kernel_size': 3, 'strides': 2, 'padding': 'same', 'use_bias': False, 'activation': 'relu'}),
        (('Dark_res', 8),
         {'filters': 512, 'kernel_size': 3, 'strides': 1, 'padding': 'same', 'use_bias': False, 'activation': 'relu'}),
    ]
    _, output_512 = block.build(output_256, structure_512, return_model=False)
    structure_1024 = [
        ('CBA',
         {'filters': 1024, 'kernel_size': 3, 'strides': 2, 'padding': 'same', 'use_bias': False, 'activation': 'relu'}),
        (('Dark_res', 4),
         {'filters': 1024, 'kernel_size': 3, 'strides': 1, 'padding': 'same', 'use_bias': False, 'activation': 'relu'}),
    ]
    _, output_1024 = block.build(output_512, structure_1024, return_model=False)
    model = keras.Model(input, (output_256, output_512, output_1024), name='Darknet53')
    return model


def yolov1(input_shape=448, split=7, number=2, classification=20):
    block_input = input_shape
    config_list = [
        ('CBA', dict(k=7, f=64, s=2, p='same')),
        ('pool', dict(pool_size=2, s=2, style='max')),

        ('CBA', dict(k=3, f=192, s=1, p='same')),
        ('pool', dict(pool_size=2, s=2, style='max')),

        ('CBA', dict(k=1, f=128, s=1, p='same')),
        ('CBA', dict(k=3, f=256, s=1, p='same')),
        ('CBA', dict(k=1, f=256, s=1, p='same')),
        ('CBA', dict(k=3, f=512, s=1, p='same')),
        ('pool', dict(pool_size=2, s=2, style='max')),
        # x4
        ('CBA', dict(k=1, f=256, s=1, p='same')),
        ('CBA', dict(k=3, f=512, s=1, p='same')),
        ('CBA', dict(k=1, f=256, s=1, p='same')),
        ('CBA', dict(k=3, f=512, s=1, p='same')),
        ('CBA', dict(k=1, f=256, s=1, p='same')),
        ('CBA', dict(k=3, f=512, s=1, p='same')),
        ('CBA', dict(k=1, f=256, s=1, p='same')),
        ('CBA', dict(k=3, f=512, s=1, p='same')),

        ('CBA', dict(k=1, f=512, s=1, p='same')),
        ('CBA', dict(k=3, f=1024, s=1, p='same')),
        ('pool', dict(pool_size=2, s=2, style='max')),
        # x2
        ('CBA', dict(k=1, f=512, s=1, p='same')),
        ('CBA', dict(k=3, f=1024, s=1, p='same')),
        ('CBA', dict(k=1, f=512, s=1, p='same')),
        ('CBA', dict(k=3, f=1024, s=1, p='same')),
        ('CBA', dict(k=3, f=1024, s=1, p='same')),
        ('CBA', dict(k=3, f=1024, s=2, p='same')),

        ('CBA', dict(k=3, f=1024, s=1, p='same')),
        ('CBA', dict(k=3, f=1024, s=1, p='same')),
        # conn layer
        ('TF', layers.Flatten()),
        ('TF', layers.Dense(units=4096, activation='relu')),
        ('TF', layers.Dense(units=split*split*(classification+number*5), activation='sigmoid')),
        ('TF', layers.Reshape(target_shape=(split, split, classification+number*5)))
    ]
    return block.build(block_input, config_list)


yolo_max_boxes = 100
yolo_iou_threshold = 0.5
yolo_score_threshold = 0.5


def yolov3(class_num):
    # CBA5 block
    def Conv_block(input, filter):
        def CUp(input, filter):
            structures = [
                ('CBA', {'filters': filter, 'kernel_size': 1, 'strides': 1, 'padding': 'same', 'use_bias': False, 'activation': 'relu'}),
                ('TF', layers.UpSampling2D(2))
            ]
            return block.build(input, structures, return_model=False)[1]

        def CBA5(input, filter):
            structures = [
                ('CBA', {'filters': filter // 2, 'kernel_size': 1, 'strides': 1, 'padding': 'same', 'use_bias': False, 'activation': 'relu'}),
                ('CBA', {'filters': filter, 'kernel_size': 3, 'strides': 1, 'padding': 'same', 'use_bias': False, 'activation': 'relu'}),
                ('CBA', {'filters': filter // 2, 'kernel_size': 1, 'strides': 1, 'padding': 'same', 'use_bias': False, 'activation': 'relu'}),
                ('CBA', {'filters': filter, 'kernel_size': 3, 'strides': 1, 'padding': 'same', 'use_bias': False, 'activation': 'relu'}),
                ('CBA', {'filters': filter // 2, 'kernel_size': 1, 'strides': 1, 'padding': 'same', 'use_bias': False, 'activation': 'relu'}),
            ]
            return block.build(input, structures, return_model=False)[1]

        if isinstance(input, tuple):
            input, deep_input = input
            deep_input = CUp(deep_input, filter // 2)
            input = layers.Concatenate()([input, deep_input])
        return CBA5(input, filter)

    # CUp+Concat block
    def CUp_Concat(input, filter):
        def CUp(input, filter):
            structures = [
                ('CBA', {'filters': filter, 'kernel_size': 1, 'strides': 1, 'padding': 'same', 'use_bias': False, 'activation': 'relu'}),
                ('TF', layers.UpSampling2D(2))
            ]
            return block.build(input, structures, return_model=False)

        res_input, cba_input = input
        _, cba_out = CUp(cba_input, filter)
        return layers.Concatenate()([res_input, cba_out])

    # CBA2 block
    def CBA2(_input, filter, class_num):
        input = layers.Input(_input.shape[1:])
        structures = [
            ('CBA', {'filters': filter, 'kernel_size': 3, 'strides': 1, 'padding': 'same', 'use_bias': False, 'activation': 'relu'}),
            ('TF', layers.Conv2D(filters=3 * (class_num + 5), kernel_size=1, strides=1, padding='same', use_bias=True)),
        ]
        model = block.build(input, structures, return_model=True)
        output = model(input)
        output = layers.Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], 3, class_num + 5)))(output)
        return tf.keras.Model(input, output, name='output%d' % filter)

    # inputlayer
    input = layers.Input([416, 416, 3])
    # Darknet53 block
    darknet53 = Darknet53()
    darknet_256, darknet_512, darknet_1024 = darknet53(input)
    # 1024 branch
    conv_1024 = Conv_block(darknet_1024, 1024)
    out_1024 = CBA2(conv_1024, 1024, class_num)(conv_1024)
    # 512 branch
    conv_512 = Conv_block((darknet_512, conv_1024), 512)
    out_512 = CBA2(conv_512, 512, class_num)(conv_512)
    # 256 branch
    conv_256 = Conv_block((darknet_256, conv_512), 256)
    out_256 = CBA2(conv_256, 256, class_num)(conv_256)
    # build modual
    return keras.Model(inputs=input, outputs=[out_256, out_512, out_1024])


def _meshgrid(n_a, n_b):
    return [
        tf.reshape(tf.tile(tf.range(n_a), [n_b]), (n_b, n_a)),
        tf.reshape(tf.repeat(tf.range(n_b), n_a), (n_b, n_a))
    ]


def yolo_boxes(pred, anchors, classes):
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    grid_size = tf.shape(pred)[1:3]
    box_xy, box_wh, objectness, class_probs = tf.split(
        pred, (2, 2, 1, classes), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    # !!! grid[x][y] == (y, x)
    grid = _meshgrid(grid_size[1], grid_size[0])
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
             tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box


def yolo_nms(outputs, anchors, masks, classes):
    # boxes, conf, type
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    scores = confidence * class_probs

    dscores = tf.squeeze(scores, axis=0)
    scores = tf.reduce_max(dscores, [1])
    bbox = tf.reshape(bbox, (-1, 4))
    classes = tf.argmax(dscores, 1)
    selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
        boxes=bbox,
        scores=scores,
        max_output_size=yolo_max_boxes,
        iou_threshold=yolo_iou_threshold,
        score_threshold=yolo_score_threshold,
        soft_nms_sigma=0.5
    )

    num_valid_nms_boxes = tf.shape(selected_indices)[0]

    selected_indices = tf.concat([selected_indices, tf.zeros(yolo_max_boxes - num_valid_nms_boxes, tf.int32)], 0)
    selected_scores = tf.concat([selected_scores, tf.zeros(yolo_max_boxes - num_valid_nms_boxes, tf.float32)], -1)

    boxes = tf.gather(bbox, selected_indices)
    boxes = tf.expand_dims(boxes, axis=0)
    scores = selected_scores
    scores = tf.expand_dims(scores, axis=0)
    classes = tf.gather(classes, selected_indices)
    classes = tf.expand_dims(classes, axis=0)
    valid_detections = num_valid_nms_boxes
    valid_detections = tf.expand_dims(valid_detections, axis=0)

    return boxes, scores, classes, valid_detections
