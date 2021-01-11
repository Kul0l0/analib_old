import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

import const


def get_file_list(dir_path):
    """
    return the list contain all data-file-path

    :param dir_path: the dir path of all data file
    :type dir_path: str
    :return: the list of all data-file-path
    :rtype: dict
    """
    if not os.path.isdir(dir_path):
        print('bad data path!')
        raise

    file_path_type_dict = {}
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        file_type = str.split(file, '.')[-1]

        if file_type in const.FILE_TYPE:
            file_path_type_dict[file_path] = file_type
            print(file_path)

    return file_path_type_dict


def csv_get_attribute(file_path):
    df = pd.read_csv(file_path, nrows=10)
    df = pd.DataFrame(df.dtypes, columns=['type'])
    df.insert(loc=0, column='name', value = df.index)

    file_name = str.split(file_path, '.')[0].split('/')[-1]
    df['file_name'] = file_name
    #df.insert(loc=-1, column='file_name', value = file_name)
    return df

def generate_dictionary(dir_path, result_path='./'):
    """
    generate the data dictionary

    :param dir_path: the dir path contain all data file
    :type dir_path: str
    :param result_path: the path save dictionary
    :type result_path: str
    """
    file_path_type_dict = get_file_list(dir_path)

    dictionary = pd.DataFrame()
    options = {
        'csv': csv_get_attribute,
    }
    for file_path, file_type in file_path_type_dict.items():
        df = options[file_type](file_path)
        dictionary = pd.concat([dictionary, df])

    dictionary.drop_duplicates(ignore_index=True, inplace=True)

    dictionary.to_csv(os.path.join(result_path, 'dictionary.csv'), index=False)

