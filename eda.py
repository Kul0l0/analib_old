import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

import const


def get_file_list(data_dir_path):
    """
    return the list contain all data-file-path

    :param data_dir_path: the dir path contain all data file
    :type data_dir_path: str
    :return: the list contain all data-file-path
    :rtype: dict
    """
    file_path_type_dict = {}
    if os.path.isdir(data_dir_path):
        for file in os.listdir(data_dir_path):
            file_path = os.path.join(data_dir_path, file)
            file_type = str.split(file, '.')[-1]

            if file_type in const.FILE_TYPE:
                file_path_type_dict[file_path] = file_type
                print(file_path)

        return file_path_type_dict
    else:
        print('bad data path!')
        raise


def csv_get_val_type_ser(file_path):
    df = pd.read_csv(file_path, nrows=10)
    return df.dtypes


def generate_dictionary(data_dir_path, result_path):
    """
    generate the data dictionary

    :param data_dir_path: the dir path contain all data file
    :type data_dir_path: str
    :param result_path: the path save dictionary
    :type result_path: str
    """
    file_path_type_dict = get_file_list(data_dir_path)

    dictionary = pd.DataFrame()
    options = {
        'csv': csv_get_val_type_ser,
    }
    for file_path, file_type in file_path_type_dict.items():
        ser = options[file_type](file_path)
        dictionary = pd.concat([dictionary, pd.DataFrame(ser, columns=['type'])], )

    dictionary.insert(loc=0, column='name', value = dictionary.index)
    dictionary.drop_duplicates(ignore_index=True, inplace=True)

    dictionary.to_csv(os.path.join(result_path, 'dictionary.csv'), index=False)

################# old code #############
########################################
def get_classifier_features_name(df):
    nuniq = df.nunique()

    return set(nuniq[nuniq < 10].index.to_list())


def get_continue_features_name(df):
    col_nobj = df.dtypes[df.dtypes != object].index.to_list()
    col_classifier = get_classifier_features_name(df)

    return set(df.columns) - col_classifier & set(col_nobj)


def plot_classifier(df, col_classifier, y=False, sample_set=False):
    nrows = len(col_classifier)
    ncols = 2 if y else 1
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))

    for i, col in enumerate(col_classifier):
        # foo = df[~df[col].isna()]
        if y:
            foo = pd.crosstab(df[sample_set], df[col], normalize='index', )
            _ = foo.plot.barh(stacked=True, ax=axs[i][0], title=col, xlim=(0, 1.3))
            foo = pd.crosstab(df[col], df[y], normalize='index', )
            _ = foo.plot.barh(stacked=True, ax=axs[i][1], title=col, xlim=(0, 1.3))
        else:
            foo = pd.crosstab(df[sample_set], df[col], normalize='index', )
            _ = foo.plot.barh(stacked=True, ax=axs[i], title=col, xlim=(0, 1.3))
    plt.subplots_adjust(hspace=0.3)


def plot_continue(df, col_continue, y=False, sample_set=False):
    nrows = len(col_continue)
    ncols = 2 if y else 1
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))
    for i, col in enumerate(col_continue):
        foo = df[~df[col].isna()]
        if y:
            _ = sns.distplot(foo[foo[sample_set] == 'TRAIN'][col], label='train', ax=axs[i][0])
            _ = sns.distplot(foo[foo[sample_set] == 'TEST'][col], label='test', ax=axs[i][0])
            axs[i][0].legend()

            _ = sns.distplot(foo[foo[y] == 1][col], label='%s 1' % y, ax=axs[i][1])
            _ = sns.distplot(foo[foo[y] == 0][col], label='%s 0' % y, ax=axs[i][1])
            axs[i][1].legend()
        else:
            _ = sns.distplot(foo[foo[sample_set] == 'TRAIN'][col], label='train', ax=axs[i])
            _ = sns.distplot(foo[foo[sample_set] == 'TEST'][col], label='test', ax=axs[i])
            axs[i].legend()


class Ana(object):
    def __init__(self, df, y=None, sample_set=None):
        # init
        self.df = df
        self.y = y
        self.sample_set = sample_set

        # 特征类型区分
        self.col_all_features = [x for x in self.df.columns if x not in [y, sample_set]]
        self.col_classifier = get_classifier_features_name(self.df[self.col_all_features])
        self.col_continue = get_continue_features_name(self.df[self.col_all_features])
        self.col_useless = set(self.col_all_features) - self.col_classifier - self.col_continue

    def features(self):
        foo = {self.y: 'target', self.sample_set: 'sample_set'}
        foo.update({x: 'classifier' for x in self.col_classifier})
        foo.update({x: 'continue' for x in self.col_continue})
        foo.update({x: 'useless' for x in self.col_useless})

        foo = pd.Series(foo, name='feature_type')
        azz = self.df.dtypes
        azz.name = 'dtype'

        return pd.concat([foo, azz], axis=1, sort=False)

    def plot(self, type, y=False):
        if y is True:
            y = self.y

        if type == 'classifier':
            plot_classifier(self.df, self.col_classifier, y, self.sample_set)

        if type == 'continue':
            plot_continue(self.df, self.col_continue, y, self.sample_set)


if __name__ == '__main__':
    data_path = '/home/hanhe/dev_e/Data/kaggle/riiid'
    # getDict(data_path)
