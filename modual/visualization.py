#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np


def imshow(ax, img):
    ax.imshow(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            text = ax.text(j, i, img[i, j], ha="center", va="center", color="w", fontsize=8)


def confusion_matrix_layout(number, label_name=None):
    fig = plt.figure(dpi=200)
    gs = fig.add_gridspec(number + 1, number + 3)
    ax0 = fig.add_subplot(gs[:number, :number])
    ax0.axes.xaxis.set_ticks([])
    ax0.set_yticks(np.arange(len(label_name)), labels=label_name)
    ax0.set_ylabel('True')
    ax1 = fig.add_subplot(gs[number, :number])
    ax1.axes.yaxis.set_ticks([])
    ax1.set_xticks(np.arange(len(label_name)), labels=label_name)
    ax1.set_xlabel('Pred')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax2 = fig.add_subplot(gs[:number, number])
    ax2.axes.xaxis.set_ticks([])
    ax2.axes.yaxis.set_ticks([])
    ax3 = fig.add_subplot(gs[:, number + 1])
    ax3.axes.xaxis.set_ticks([])
    ax3.axes.yaxis.set_ticks([])
    ax3.set_xlabel('recall')
    ax4 = fig.add_subplot(gs[:, number + 2])
    ax4.axes.xaxis.set_ticks([])
    ax4.axes.yaxis.set_ticks([])
    ax4.set_xlabel('precision')
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    # plt.tight_layout()
    return fig, ax0, ax1, ax2, ax3, ax4


if __name__ == '__main__':
    N = 5
    label = list('abcde')
    y_true = np.random.randint(0, N, 1000)
    y_pred = np.random.randint(0, N, 1000)
    foo = confusion_matrix_layout(N, label)
    imshow(foo[1], )
    plt.show()
