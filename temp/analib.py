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


def generate_dictionary(base_df=None, file_path=None, output_path='./', save=True, useless_col=None):
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
    def attach_label(ser, label_dict):
        for label, val in label_dict.items():
            if val and ser in val:
                return label
        return None

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
    # build return df
    dictionary = pd.DataFrame(data=base_df.dtypes, columns=[const.CN_TYPE])
    dictionary.insert(loc=0, column=const.CN_FEATURE_NAME, value = dictionary.index)
    ## add label
    useless_col_dict = {const.MSG_USELESS: useless_col}
    dictionary[const.CN_FEATURE_LABEL] = dictionary[const.CN_FEATURE_NAME].apply(func=attach_label, args=[useless_col_dict])
    dictionary[const.CN_EXPLANATION] = None
    # drop useless col
    use_col = set(base_df.columns)-set(useless_col)
    useful_base_df = base_df[use_col]
    # describe
    describe_df = useful_base_df.describe().T
    # nunique
    nuniq = useful_base_df.nunique()
    nuniq.name = const.CN_NUNIQUE
    # missing rate
    missing_rate = useful_base_df.isna().sum()/useful_base_df.shape[0]
    missing_rate.name = const.CN_MISSING_RATE
    # concat
    dictionary = pd.concat([dictionary, nuniq, missing_rate, describe_df], axis=1)
    # output check
    if save:
        if file_path:
            output_name = file_path.split('.')[0].split('/')[-1]
        else:
            output_name = const.OP_DEFAULT_FILE_NAME
        dictionary.to_csv(os.path.join(output_path, '%s_dict.csv'%output_name), index=False)
    else:
        return dictionary

def write_excel(file_path, extra='', sheet_names=[], sheet_contents=[]):
    with pd.ExcelWriter("%s%s.xlsx"%(file_path, extra)) as ew:
        for sheet_name, df in zip(sheet_names, sheet_contents):
            df.to_excel(excel_writer=ew, sheet_name=sheet_name, index=False)

def add_colunms(table:pd.DataFrame, locs, col_names, values):
    for loc, col, val in zip(locs, col_names, values):
        if table.shape[0] == 0:
            val = [val]
        table.insert(loc=loc, column=col, value=val)
    return table

def paint_table(table, key, colors):
    pass


class EDA(object):

    def __init__(self, data_dir_path=None, output_path=None, useless_col=None):
        self.data_dir_path = data_dir_path
        self.output_path = output_path
        self.useless_col = useless_col
        self.get_df = {
            'csv': pd.read_csv,
        }
        self.df_list = []
        # self.sheet_names = []
        # self.sheet_contents = []

    @staticmethod
    def value_count_df(ser, bins=None):
        count_ser = ser.value_counts(sort=False, bins=bins, normalize=False, dropna=False)
        count_ser.name = const.CN_COUNT
        percent_ser = ser.value_counts(sort=False, bins=bins, normalize=True, dropna=False)
        percent_ser.name = const.CN_PERCENT
        # build result
        return_df = pd.concat([count_ser, percent_ser], axis=1)
        return_df.index.name = const.CN_VALUE
        return_df.reset_index(inplace=True)
        return_df.insert(loc=0, column=const.CN_FEATURE_NAME, value=ser.name)
        # return
        return return_df

    def generate_distribution(self, data_df, col_names, data_types):
        # The types that have been processed include int, float, bool, object
        return_df = pd.DataFrame()
        for idx, col_name, data_type in zip(range(len(col_names)), col_names, data_types):
            nunique = data_df[col_name].nunique()
            bins = None
            if data_type in (int, float, bool):
                if nunique > const.DB_INT_DIVIDE:
                    ser, bins = pd.cut(data_df[col_name], bins=const.BIN_NUMBER, retbins=True)
                distribution_df = EDA.value_count_df(data_df[col_name], bins=bins)
            elif data_type == object:
                if nunique > const.DB_OBJECT_DIVIDE:
                    distribution_df = add_colunms(
                        table=pd.DataFrame(),
                        locs=[0,1],
                        col_names=[const.CN_FEATURE_NAME, const.CN_VALUE],
                        values=[col_name, const.MSG_MORE_VALUE]
                    )
                else:
                    distribution_df = EDA.value_count_df(data_df[col_name], bins=None)
            else:
                distribution_df = add_colunms(
                    table=pd.DataFrame(),
                    locs=[0, 1],
                    col_names=[const.CN_FEATURE_NAME, const.CN_VALUE],
                    values=[col_name, const.MSG_UNTREATED]
                )
            distribution_df.insert(loc=1, column=const.CN_TYPE, value=data_type)
            distribution_df.sort_values(by=const.CN_VALUE, inplace=True)
            # paint
            if idx%2:
                distribution_df.style.set_properties(**{'background-color': 'aliceblue'})
            # build return
            return_df = pd.concat([return_df, distribution_df])

        # fillna in bins
        return_df[const.CN_VALUE].fillna('NaN', inplace=True)
        return return_df

    def base_EDA(self):
        type_paths = get_files_type_paths(self.data_dir_path)
        for file_type, file_paths in type_paths.items():
            # check the type of files
            if file_type not in const.DATA_FILE_TYPE:
                continue
            for file_path in file_paths:
                sheet_contents = []
                sheet_names = []
                # load data to DataFrame
                df = self.get_df[file_type](file_path)
                self.df_list.append(df)
                file_name = file_path.split('.')[0].split('/')[-1]
                # generate dictionary
                dict_df = generate_dictionary(base_df=df, save=False, useless_col=self.useless_col)
                sheet_names.append("Dictionary")
                sheet_contents.append(dict_df)
                # drop useless feature columns
                # if self.useless_col:
                #     dict_df.drop(index=self.useless_col, inplace=True, errors='ignore')
                # generate distribution
                distribution_df = self.generate_distribution(
                    data_df=df,
                    col_names=dict_df[const.CN_FEATURE_NAME],
                    data_types=dict_df[const.CN_TYPE],
                )
                sheet_names.append("Distribution")
                sheet_contents.append(distribution_df)
                # write excel
                write_excel(
                    file_path=os.path.join(self.output_path, file_name),
                    extra=const.OP_EDA_EXTRA,
                    sheet_names=sheet_names,
                    sheet_contents=sheet_contents,
                )













