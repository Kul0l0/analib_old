import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

import const


def get_files_type_paths(home_path):
    """
    return a dict contain all data-file-paths and file types.
    the keys is file-type.
    the values is a list of file path.

    :param home_path: a dir path of all data file
    :type home_path: str
    :return: a dict of all data-file-path
    :rtype: dict
    """
    if not os.path.isdir(home_path):
        print('bad dir path!')
        raise

    file_type_path_dict = {}
    for dir_path, _, files_name in os.walk(home_path, topdown=True):
        for file_name in files_name:
            file_path = os.path.join(dir_path, file_name)
            file_type = str.split(file_name, '.')[-1]

            if file_type_path_dict.get(file_type) is None:
                file_type_path_dict[file_type] = [file_path]
            file_type_path_dict[file_type].append(file_path)

    return file_type_path_dict


def csv_get_attribute(file_path):
    """
    return a DataFrame that contain dtypes of the provided data

    :param file_path: csv data file path
    :type file_path: str
    :return: a DataFrame that contain dtypes
    :rtype: DataFrame
    """
    df = pd.read_csv(file_path, nrows=const.NROWS)
    df = pd.DataFrame(df.dtypes, columns=['type'])
    df.insert(loc=0, column='name', value = df.index)

    file_name = str.split(file_path, '.')[0].split('/')[-1]
    df['file_name'] = file_name
    #df.insert(loc=-1, column='file_name', value = file_name)
    return df

def generate_dictionary(dir_path, result_path='./', save=True):
    """
    generate a data dictionary.
    save to the result path or return directly.

    :param dir_path: a dir path contain all data file
    :type dir_path: str
    :param result_path: the path save dictionary
    :type result_path: str
    :param save: if save=True save a file, else return the dictionary
    :type save: bool
    :return: a Dataframe of dictionary
    :rtype: DataFrame
    """
    file_type_paths = get_files_type_paths(dir_path)

    dictionary = pd.DataFrame()
    options = {
        'csv': csv_get_attribute,
    }
    for file_type, file_paths in file_type_paths.items():
        if file_type not in const.DATA_FILE_TYPE:
            continue
        for file_path in file_paths:
            df = options[file_type](file_path)
            dictionary = pd.concat([dictionary, df])

    dictionary.drop_duplicates(ignore_index=True, inplace=True)

    if save:
        dictionary.to_csv(os.path.join(result_path, 'dictionary.csv'), index=False)
    else:
        return dictionary


class EDA(object):
    def __init__(self, dir_path=None):

        # working
        type_paths = get_files_type_paths(dir_path)











































