#!/usr/bin/env python
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from . import visualization as vz
import os


def cal_mean(func, y_true, y_pred):
    score = func(y_true, y_pred, average=None)
    score = np.append(score, np.mean(score))
    return np.expand_dims(score, -1).round(2)


def confusion_matrix(y_true: np.ndarray, y_pred_prob: np.ndarray, label_name=None, outdir=None):
    y_pred = np.argmax(y_pred_prob, axis=1)
    fig, *axes = vz.confusion_matrix_layout(len(label_name), label_name)
    # confusion matrix
    cm = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
    vz.imshow(axes[0], cm)
    # sum simple
    pred_sum = cm.sum(axis=0, keepdims=True)
    vz.imshow(axes[1], pred_sum)
    true_sum = cm.sum(axis=1, keepdims=True)
    vz.imshow(axes[2], true_sum)
    # recall and precision
    recall = cal_mean(metrics.recall_score, y_true, y_pred)
    vz.imshow(axes[3], recall)
    precision = cal_mean(metrics.precision_score, y_true, y_pred)
    vz.imshow(axes[4], precision)
    # accuracy
    accuracy = metrics.accuracy_score(y_true, y_pred)
    axes[0].set_title('Accuracy: %f' % accuracy)
    plt.yticks(rotation=0)
    if outdir:
        plt.savefig(fname=os.path.join(outdir, 'confusion_matrix.png'))
    else:
        plt.show()


def AUC(y_true: np.ndarray, y_pred_prob: np.ndarray):
    pass


def main():
    y_true = np.random.randint(0, 10, 10000)
    y_pred = np.random.randint(0, 10, 10000)
    print(y_true)
    print(y_pred)
    name = list('abcde')
    confusion_matrix(y_true, y_pred, name)


if __name__ == '__main__':
    main()
