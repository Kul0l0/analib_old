#!/usr/bin/env python
# encoding: utf-8
"""
@Author  : Kul0l0
@File    : prepare.py
@Time    : 2021/3/24 上午12:14
"""
import tensorflow as tf
import cv2
import os
from tqdm import tqdm
import numpy as np

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def build_decoder(with_labels=True, target_size=(224, 224), ext='jpg'):
    def decode(path):
        file_bytes = tf.io.read_file(path)
        if ext == 'png':
            img = tf.image.decode_png(file_bytes, channels=3)
        elif ext in ['jpg', 'jpeg']:
            img = tf.image.decode_jpeg(file_bytes, channels=3)
        else:
            raise ValueError("Image extension not supported")

        img = tf.cast(img, tf.float32) / 255.0
        img = tf.image.resize(img, target_size)

        return img

    def decode_with_labels(path, label):
        return decode(path), label

    return decode_with_labels if with_labels else decode


def build_augmenter(with_labels=True):
    def augment(img):
        img = tf.image.random_flip_left_right(img)
        # img = tf.image.random_flip_up_down(img)
        return img

    def augment_with_labels(img, label):
        return augment(img), label

    return augment_with_labels if with_labels else augment


def build_dataset(paths, labels=None, bsize=32, cache=True,
                  decode_fn=None, augment_fn=None,
                  augment=True, repeat=True, shuffle=1024,
                  cache_dir=""):
    if cache_dir != "" and cache is True:
        os.makedirs(cache_dir, exist_ok=True)

    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)

    if augment_fn is None:
        augment_fn = build_augmenter(labels is not None)

    auto = tf.data.experimental.AUTOTUNE
    slices = paths if labels is None else (paths, labels)

    dataset = tf.data.Dataset.from_tensor_slices(slices)
    dataset = dataset.map(decode_fn, num_parallel_calls=auto)
    dataset = dataset.cache(cache_dir) if cache else dataset
    dataset = dataset.map(augment_fn, num_parallel_calls=auto) if augment else dataset
    dataset = dataset.repeat() if repeat else dataset
    dataset = dataset.shuffle(shuffle) if shuffle else dataset
    dataset = dataset.batch(bsize).prefetch(auto)

    return dataset


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    if not isinstance(value, (list, np.ndarray)):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    if not isinstance(value, (list, np.ndarray)):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    if not isinstance(value, (list, np.ndarray)):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _tensor_feature(value):
    """Returns an bytes_list from a tensor"""
    value = tf.io.serialize_tensor(value)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.numpy()]))


def center_crop(img):
    x, y = img.shape[0], img.shape[1]
    start = abs(x - y) // 2
    if x > y:
        return img[start:y + start, :]
    else:
        return img[:, start:x + start]


def serialize_example(img, label=None):
    feature = {'image': _bytes_feature(img)}
    if label is not None:
        feature['label'] = _float_feature(label)
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def generate_tfr(file_path_list, label_list, out_file_name, image_size=None, color=True, crop=False):
    # if color is True, img shape is [x,x,3]. else is [x,x]
    with tf.io.TFRecordWriter(out_file_name) as writer:
        for idx, file_path in tqdm(enumerate(file_path_list)):
            img = cv2.imread(file_path, color)
            # img = img[:, :, tf.newaxis] if len(img.shape)==2 else img
            img = center_crop(img) if crop else img
            img = cv2.resize(img, (image_size, image_size))
            if label_list is None:
                example = serialize_example(img.tobytes(), None)
            else:
                example = serialize_example(img.tobytes(), label_list[idx])
            writer.write(example)
