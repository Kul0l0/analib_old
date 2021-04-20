#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : Kul0l0
@File    : prepare.py
@Time    : 2021/3/24 上午12:14
'''
import tensorflow as tf
import cv2
import os
from tqdm import tqdm
import numpy as np
import numpy as np
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    if not isinstance(value, (list,np.ndarray)):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    if not isinstance(value, (list,np.ndarray)):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    if not isinstance(value, (list,np.ndarray)):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _tensor_feature(value):
    """Returns an bytes_list from a tensor"""
    value = tf.io.serialize_tensor(value)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.numpy()]))

def center_crop(img):
    x,y = img.shape[0], img.shape[1]
    start = abs(x-y)//2
    if x > y:
        return img[start:y+start, :]
    else:
        return img[:, start:x+start]

def save_TFR(data, out_file_name):
    pass


def serialize_example(img, label=None):
    if label is None:
        feature = {
            'image': _bytes_feature(img),
        }
    else:
        feature = {
            'image': _bytes_feature(img),
            'label': _float_feature(label),
        }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def resize_save_TFR(file_list, label_list, out_file_name, image_size, color=True, type='crop'):
    with tf.io.TFRecordWriter(out_file_name) as writer:
        for idx,file in tqdm(enumerate(file_list)):
            img = cv2.imread(file, color)
            img = center_crop(img)
            img = cv2.resize(img, image_size)
            if not color:
                img = img[:,:,tf.newaxis]
            if label_list is not None:
                example = serialize_example(img.tobytes(), label_list[idx])
            else:
                example = serialize_example(img.tobytes(), None)
            writer.write(example)



