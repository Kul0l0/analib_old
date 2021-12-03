#!/usr/bin/env python
import numpy as np
from sklearn import metrics, preprocessing
import matplotlib.pyplot as plt
from . import visualization as vz
# import visualization as vz
import os


def cal_mean(func, y_true, y_pred):
    score = func(y_true, y_pred, average=None)
    score = np.append(score, np.mean(score))
    return np.expand_dims(score, -1).round(2)


def confusion_matrix(y_true: np.ndarray, y_pred_prob: np.ndarray, label_name=None, outdir=None):
    y_pred_class = np.argmax(y_pred_prob, axis=1).ravel()
    fig, *axes = vz.confusion_matrix_layout(len(label_name), label_name)
    # confusion matrix
    cm = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred_class)
    vz.imshow(axes[0], cm)
    # sum simple
    pred_sum = cm.sum(axis=0, keepdims=True)
    vz.imshow(axes[1], pred_sum)
    true_sum = cm.sum(axis=1, keepdims=True)
    vz.imshow(axes[2], true_sum)
    # recall and precision
    recall = cal_mean(metrics.recall_score, y_true, y_pred_class)
    vz.imshow(axes[3], recall)
    precision = cal_mean(metrics.precision_score, y_true, y_pred_class)
    vz.imshow(axes[4], precision)
    # accuracy
    accuracy = metrics.accuracy_score(y_true, y_pred_class)
    axes[0].set_title('Accuracy: %f' % accuracy)
    plt.yticks(rotation=0)
    if outdir:
        plt.savefig(fname=os.path.join(outdir, 'confusion_matrix.png'))
    else:
        plt.show()


def PR_AUC(y_true: np.ndarray, y_pred_prob: np.ndarray, label_number: int, outdir=None):
    fig = plt.figure(dpi=200)
    y_true_binary = preprocessing.label_binarize(y_true, classes=range(label_number))
    recall, precision, auc = dict(), dict(), dict()
    lw = 2
    for label in range(label_number):
        precision[label], recall[label], thresholds = metrics.precision_recall_curve(y_true_binary[:, label], y_pred_prob[:, label])
        auc[label] = metrics.auc(recall[label], precision[label])
    # micro auc
    precision["micro"], recall["micro"], _ = metrics.precision_recall_curve(y_true_binary.ravel(), y_pred_prob.ravel())
    auc["micro"] = metrics.auc(recall["micro"], precision["micro"])
    # plot
    for label, v in recall.items():
        plt.plot(recall[label], precision[label], label="label: %s, AUC: %f" % (label, auc[label]), lw=lw)
    plt.plot([1, 0], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    if outdir:
        plt.savefig(fname=os.path.join(outdir, 'PR_AUC.png'))
    else:
        plt.show()


def ROC_AUC(y_true: np.ndarray, y_pred_prob: np.ndarray, label_number: int, outdir=None):
    fig = plt.figure(dpi=200)
    y_true_binary = preprocessing.label_binarize(y_true, classes=range(label_number))
    fpr, tpr, auc = dict(), dict(), dict()
    lw = 2
    for label in range(label_number):
        fpr[label], tpr[label], thresholds = metrics.roc_curve(y_true_binary[:, label], y_pred_prob[:, label])
        auc[label] = metrics.auc(fpr[label], tpr[label])
    # macro_auc
    macro_auc = metrics.roc_auc_score(y_true_binary, y_pred_prob)
    # micro auc
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_true_binary.ravel(), y_pred_prob.ravel())
    auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
    # plot
    for label, v in fpr.items():
        plt.plot(fpr[label], tpr[label], label="label: %s, AUC: %f" % (label, auc[label]), lw=lw)
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(np.mean(macro_auc))
    plt.legend(loc="lower right")
    if outdir:
        plt.savefig(fname=os.path.join(outdir, 'ROC_AUC.png'))
    else:
        plt.show()


def main():
    N = 5
    y_true = np.random.randint(0, N, 100).flatten()
    y_pred = np.random.randn(100, N)
    name = list('abcde')
    ROC_AUC(y_true, y_pred, label_number=N)
    # confusion_matrix(y_true, y_pred, name)


if __name__ == '__main__':
    main()
