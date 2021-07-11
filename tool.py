#!/usr/bin/env python
# encoding: utf-8
"""
@Author  : Kul0l0
@File    : tool.py
@Time    : 2021/7/8 下午10:08
"""

from pandas import DataFrame
import os
from tqdm import tqdm
import xmltodict
import cv2 as cv


def draw_bounding_box(img, pts, text):
    """
    draw bounding box on input image
    :param img: image path or numpy array of image
    :type img: str or numpy array
    :param pts: (pt1, pt2), Vertexes of the rectangle.
    :type pts: iterable
    :param text: text of bounding box
    :type text: str
    :return: image with drawn bounding box
    :rtype: numpy array
    """
    if isinstance(img, str):
        img = cv.imread(img)
    pt1, pt2 = pts
    cv.rectangle(
        img=img,
        pt1=pt1,
        pt2=pt2,
        color=(0, 0, 255),
        thickness=3,
    )
    cv.putText(
        img=img,
        text=text,
        org=pt1,
        fontFace=cv.FONT_HERSHEY_SIMPLEX,
        fontScale=2,
        color=(0, 0, 255),
        thickness=2,
    )
    return img


def xml_to_dict(file_path):
    """
    接收xml文件路径,返回转化后的dict
    :param file_path: xml file path
    :type file_path: str
    :return: dict of xml data
    :rtype: dict
    """
    with open(file_path, 'r') as file:
        xml = file.read()
    return xmltodict.parse(xml)


def xml_to_df(home_path):
    def smooth(prefix, xml_d):
        for key in list(xml_d.keys()):
            if isinstance(xml_d[key], dict):
                smooth(None, xml_d[key])
                pre_now = prefix + '_' + key if prefix else key
                for k, v in xml_d.pop(key).items():
                    xml_d[pre_now + '_' + k] = v

    def split_obj(xml_d):
        res = []
        for key in list(xml_d.keys()):
            if isinstance(xml_d[key], list):
                objects = xml_d.pop(key)
                for obj in objects:
                    d = xml_d.copy()
                    d[key] = obj
                    smooth(None, d)
                    res.append(d)
        return res if res != [] else [xml_d]

    series_list = []
    files_list = os.listdir(home_path)
    for file_name in tqdm(files_list):
        file_path = home_path + file_name
        with open(file_path, 'r') as file:
            data = file.read()
        xml_dict = xmltodict.parse(data)['annotation']
        smooth(None, xml_dict)
        xml_d_l = split_obj(xml_dict)
        series_list += xml_d_l
    return DataFrame(series_list)

# home_path = '/home/hanhe/dev_e/Data/VOC2012/Annotations/'
# foo = xml_to_df(home_path)
# foo.to_csv('~/temp/annotation.csv', index=False)
# print(foo.columns)
