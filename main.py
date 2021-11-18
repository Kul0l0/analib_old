#!/usr/bin/env python
# encoding: utf-8
"""
@Author  : Kul0l0
@File    : main.py
@Time    : 2021/6/7 上午8:05
"""
from modual import template, experiment, visualization
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, datasets
import numpy as np
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def load_yolo():
    class_num = 80
    weight_path = '/home/hanhe/dev_e/Data/00.Model/yolov3.tf'
    yolo = template.yolov3(class_num)
    yolo.summary()
    keras.utils.plot_model(yolo)
    return yolo.load_weights(weight_path).expect_partial()


def augment(img, label):
    img = tf.cast(img, tf.float32)
    # img = tf.image.random_flip_left_right(img)
    # img = tf.image.resize_with_crop_or_pad(img, target_height=40, target_width=40)
    # img = tf.image.random_crop(img, [128,32,32,3])
    # img = tf.image.random_crop(img, [32,32,3])
    return img, label


if __name__ == '__main__':
    N = 1
    experiment_config = dict(
        name='ep_test',
        outdir='/home/hanhe/temp/ep_test',
        label_name=['plane', 'mobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    )
    model_config = dict(
        name='Plain',
        code='123',
        input_shape=32,
        plot=True,
        model_structure=(
            dict(name='CBA', times=1, args=dict(f=16, k=3, s=1, p='same', a='relu')),
            dict(name='res_plain', times=2 * N, args=dict(f=16, k=3, s=1, p='same', a='relu')),
            dict(name='res_plain', times=2 * N, args=dict(f=16, k=3, s=1, p='same', a='relu')),
            dict(name='res_plain', times=1, args=dict(f=32, k=3, s=2, p='same', a='relu')),
            dict(name='res_plain', times=2 * N - 1, args=dict(f=32, k=3, s=1, p='same', a='relu')),
            dict(name='res_plain', times=1, args=dict(f=64, k=3, s=2, p='same', a='relu')),
            dict(name='res_plain', times=2 * N, args=dict(f=64, k=3, s=1, p='same', a='relu')),
            dict(name='TF', times=1, args=layers.GlobalAveragePooling2D(name='GAPool')),
            dict(name='TF', times=1, args=layers.Dense(units=10, activation='softmax', name='output')),
        )
    )
    strategy = dict(
        kfold=2,
        seed=369,
        optimizer='adam',
        metrics=('accuracy'),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    )
    fit_config = dict(
        epochs=5,
        batch_size=128,
    )
    metrics = {'confusion_matrix'}
    ep = experiment.experiment(
        experiment_config=experiment_config,
        model_config=model_config,
        strategy=strategy,
        fit_config=fit_config,
        metrics=metrics,
    )
    (train_images, train_labels), (val_images, val_labels) = datasets.cifar10.load_data()
    data = np.concatenate([train_images, val_images]) / 255.0
    label = np.concatenate([train_labels, val_labels])
    ep.train((data, label), augment=augment)
