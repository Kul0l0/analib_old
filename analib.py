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


def generate_dictionary(base_df=None, file_path=None, output_path='./', save=True):
    """
    generate a data dictionary.
    save to the result path or return directly.

    :param base_df: a DataFrame
    :type base_df: DataFrame
    :param file_path: a dir path contain all data file
    :type file_path: str
    :param output_path: the path save dictionary
    :type output_path: str
    :param save: if save=True save a file, else return the dictionary
    :type save: bool
    :return: a Dataframe of dictionary
    :rtype: DataFrame
    """
    # input check
    if base_df is None and file_path is None:
        print('Please, provide a df or a path of data')
        raise

    # working
    get_df = {
        'csv': pd.read_csv,
    }
    if base_df is None:
        file_type = file_path.split('.')[-1]
        base_df = get_df[file_type](file_path)

    dictionary = pd.DataFrame(base_df.dtypes, columns=['Type'])
    dictionary.insert(loc=0, column='Name', value = dictionary.index)
    dictionary['Explanation'] = None

    # output check
    if save:
        if file_path:
            output_name = file_path.split('.')[0].split('/')[-1]
        else:
            output_name = 'temp'
        dictionary.to_csv(os.path.join(output_path, '%s_dict.csv'%output_name), index=False)
    else:
        return dictionary

def write_excel(file_path, extra='', sheet_names=[], sheet_contents=[]):
    with pd.ExcelWriter("%s%s.xlsx"%(file_path, extra)) as ew:
        for sheet_name, df in zip(sheet_names, sheet_contents):
            df.to_excel(excel_writer=ew, sheet_name=sheet_name, index=False)

class EDA(object):

    def __init__(self, data_dir_path=None, output_path=None):
        self.data_dir_path = data_dir_path
        self.output_path = output_path
        self.get_df = {
            'csv': pd.read_csv,
        }
        self.df_list = []
        self.sheet_names = []
        self.sheet_contents = []
        # self.base_EDA()


    def base_EDA(self):
        type_paths = get_files_type_paths(self.data_dir_path)
        for file_type, file_paths in type_paths.items():
            if file_type not in const.DATA_FILE_TYPE:
                continue
            for file_path in file_paths:
                df = self.get_df[file_type](file_path)
                self.df_list.append(df)
                file_name = file_path.split('.')[0].split('/')[-1]
                # dictionary
                dict_df = generate_dictionary(base_df=df, save=False)
                self.sheet_names.append("Dictionary")
                self.sheet_contents.append(dict_df)
                # write excel
                write_excel(
                    file_path=os.path.join(self.output_path, file_name),
                    extra='_EDA',
                    sheet_names=self.sheet_names,
                    sheet_contents=self.sheet_contents
                )


























































