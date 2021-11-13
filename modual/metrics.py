import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def cal_confusion_matrix(true_label: np.ndarray, pred_label: np.ndarray, label_name=None, out_path=None):
    if true_label.shape == pred_label.shape:
        cm = confusion_matrix(y_true=true_label, y_pred=pred_label, normalize='true')
        ax = sns.heatmap(data=cm, annot=True)
        ax.set_xlabel('Pred')
        ax.set_ylabel('True')
        if label_name:
            ax.set_xticklabels(label_name)
            ax.set_yticklabels(label_name)
        plt.show()
    else:
        # the shape is different
        pass


if __name__ == '__main__':
    true = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
    pred = np.random.randint(0, 3, 10)
    print(true)
    print(pred)
    name = ['apple', 'beer', 'cat']
    cal_confusion_matrix(true, pred, name)
