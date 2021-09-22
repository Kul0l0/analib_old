#!/usr/bin/env python
# encoding: utf-8
"""
@Author  : Kul0l0
@File    : detect.py
@Time    : 2021/9/20 下午7:54
"""
import sys
import cv2


def draw_boxs(img, boxs, box_color=(0, 0, 255), thickness=1, label_color=(0, 0, 0)):
    for box in boxs:
        x1, y1, = box[:2]
        x2, y2, = box[2:4]
        label = None if len(box) == 4 else box[4]
        cv2.rectangle(
            img=img,
            pt1=(x1, y1),
            pt2=(x2, y2),
            color=box_color,
            thickness=thickness,
        )
        if label:
            cv2.putText(
                img=img,
                text=label,
                org=(x1, y1),
                color=label_color,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                thickness=thickness,
            )


def mark_video(video, boxs_df):
    fps = video.get(cv2.CAP_PROP_FPS)
    while video.isOpened():
        ret, frame = video.read()
        if ret is True:
            frame_number = video.get(cv2.CAP_PROP_POS_FRAMES)
            print(frame_number)
            sub_df = boxs_df[boxs_df['frame'] == frame_number]
            draw_boxs(frame, sub_df.iloc[:, 1:].to_numpy())
            cv2.imshow('Frame', frame)
            key = cv2.waitKey(int(100))

            if key == ord('q'):
                break
        else:
            break
    video.release()
    cv2.destroyAllWindows()
