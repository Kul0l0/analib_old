#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : Kul0l0
@File    : yolo.py
@Time    : 2021/6/7 上午8:05
'''
from tensorflow import keras
import template
from tensorflow import keras

out_shape = 3*(80+5)
weight_path = '/home/hanhe/dev_e/Data/00.Model/yolov3.tf'
yolo = template.yolov3(out_shape)
yolo.summary()
#keras.utils.plot_model(yolo)
# yolo.load_weights(weight_path).expect_partial()

