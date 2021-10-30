#!/usr/bin/env python
# encoding: utf-8
"""
@Author  : Kul0l0
@File    : main.py
@Time    : 2021/6/7 上午8:05
"""
from modual import template, experiment
from tensorflow import keras


def load_yolo():
    class_num = 80
    weight_path = '/home/hanhe/dev_e/Data/00.Model/yolov3.tf'
    yolo = template.yolov3(class_num)
    yolo.summary()
    keras.utils.plot_model(yolo)
    return yolo.load_weights(weight_path).expect_partial()


if __name__ == '__main__':
    config = dict(
        input_shape=24,
        plot=True,
        model_config=(
            dict(times=3, name='CB', args=dict(f=6, k=3, s=1, p='same')),
            [
                dict(times=1, name='CB', args=dict(f=6, k=3, s=6, p='same')),
                (
                    dict(times=1, name='CB', args=dict(f=6, k=3, s=3, p='same')),
                    dict(times=1, name='CB', args=dict(f=6, k=3, s=2, p='same')),
                ),
            ],
            dict(times=1, name='CB', args=dict(f=6, k=3, s=2, p='same')),
        )
    )
    em = experiment.experiment(model_config=config)
    print(em.model.summary())
