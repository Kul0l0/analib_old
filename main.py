#!/usr/bin/env python
# encoding: utf-8
"""
@Author  : Kul0l0
@File    : main.py
@Time    : 2021/6/7 上午8:05
"""
from modual import template, experiment
from tensorflow import keras
from tensorflow.keras import layers


def load_yolo():
    class_num = 80
    weight_path = '/home/hanhe/dev_e/Data/00.Model/yolov3.tf'
    yolo = template.yolov3(class_num)
    yolo.summary()
    keras.utils.plot_model(yolo)
    return yolo.load_weights(weight_path).expect_partial()


if __name__ == '__main__':
    N = 1
    config = dict(
        input_shape=32,
        code='123',
        outpath='./temp',
        plot=True,
        name='Plain',
        model_config=(
            dict(name='CBA',       times=1,     args=dict(f=16, k=3, s=1, p='same', a='relu')),
            dict(name='res_plain', times=2*N,   args=dict(f=16, k=3, s=1, p='same', a='relu')),
            dict(name='res_plain', times=2*N,   args=dict(f=16, k=3, s=1, p='same', a='relu')),
            dict(name='res_plain', times=1,     args=dict(f=32, k=3, s=2, p='same', a='relu')),
            dict(name='res_plain', times=2*N-1, args=dict(f=32, k=3, s=1, p='same', a='relu')),
            dict(name='res_plain', times=1,     args=dict(f=64, k=3, s=2, p='same', a='relu')),
            dict(name='res_plain', times=2*N,   args=dict(f=64, k=3, s=1, p='same', a='relu')),
            dict(name='TF',        times=1,     args=layers.GlobalAveragePooling2D(name='GAPool')),
            dict(name='TF',        times=1,     args=layers.Dense(units=10, activation='softmax')),
        )
    )
    strategy = dict(
        cv=0.2,
        epochs=120,
        batch_size=128,
        optimizer='adam',
        metrics=('accuracy', 'AUC'),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    )
    ep = experiment.experiment(model_config=config)
    print(ep.model.summary())
