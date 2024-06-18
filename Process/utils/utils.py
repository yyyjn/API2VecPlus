"""
@File         :   utils.py
@Author       :   yjn
@Contact      :   yinjunnan1@gmail.com
@Version      :   1.0
@Desc         :   工具包
"""

import pickle
import os
import json
import random
from math import floor

import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def split_data(datas, split_ratio=0.8, seed=7):
    random.seed(seed)
    length = len(datas)
    indexes = list(range(length))
    random.shuffle(indexes)

    first_length = floor(length * split_ratio)
    indexes_first = indexes[:first_length]
    indexes_second = indexes[first_length:]

    data_first = [datas[i] for i in indexes_first]
    data_second = [datas[i] for i in indexes_second]

    return data_first, data_second


def split_data_by_name_raw(datas, data_names):
    train_datas, val_datas, test_datas = [], [], []
    train_names, val_names, test_names = data_names

    for data in datas:
        if data[0] in train_names:
            train_datas.append(data)
        elif data[0] in val_names:
            val_datas.append(data)
        elif data[0] in test_names:
            test_datas.append(data)

    return train_datas, val_datas, test_datas


def split_data_by_name_concate(datas, data_names):
    train_datas, val_datas, test_datas = [], [], []
    train_names, val_names, test_names = data_names

    for data in datas:
        if data[0] in train_names:
            if len(data[1]) == 30:
                train_datas.append(data)
            else:
                for i in range(0, 30 - len(data[1])):
                    data[1].append(["0"])
                train_datas.append(data)
        elif data[0] in val_names:
            if len(data[1]) == 30:
                val_datas.append(data)
            else:
                for i in range(0, 30 - len(data[1])):
                    data[1].append(["0"])
                val_datas.append(data)
        elif data[0] in test_names:
            if len(data[1]) == 30:
                test_datas.append(data)
            else:
                for i in range(0, 30 - len(data[1])):
                    data[1].append(["0"])
                test_datas.append(data)

    return train_datas, val_datas, test_datas


def split_data_by_name_remove(datas, data_names):
    train_datas, val_datas, test_datas = [], [], []
    train_names, val_names, test_names = data_names

    for data in datas:
        if data[0] in train_names:
            if len(data[1]) >= 30:
                train_datas.append(data)
            else:
                for i in range(0, 30 - len(data[1])):
                    data[1].append(["0"])
                train_datas.append(data)
        elif data[0] in val_names:
            if len(data[1]) >= 30:
                val_datas.append(data)
            else:
                for i in range(0, 30 - len(data[1])):
                    data[1].append(["0"])
                val_datas.append(data)
        elif data[0] in test_names:
            if len(data[1]) >= 30:
                test_datas.append(data)
            else:
                for i in range(0, 30 - len(data[1])):
                    data[1].append(["0"])
                test_datas.append(data)

    return train_datas, val_datas, test_datas


def split_data_by_name1(datas, data_names):
    train_datas, val_datas, test_datas = [], [], []
    train_names, val_names, test_names = data_names

    for data in datas:
        if data[0] in train_names:
            for api_path in data[1]:
                _data = (data[0], api_path, data[2], data[3])
                train_datas.append(_data)
        elif data[0] in val_names:
            for api_path in data[1]:
                _data = (data[0], api_path, data[2], data[3])
                val_datas.append(_data)
        elif data[0] in test_names:
            for api_path in data[1]:
                _data = (data[0], api_path, data[2], data[3])
                test_datas.append(_data)
            # test_datas.append(data)

    return train_datas, val_datas, test_datas


def split_data_by_name2(datas, data_names):
    train_datas, val_datas, test_datas = [], [], []
    train_names, val_names, test_names = data_names

    for data in datas:
        if data[0] in train_names:
            train_datas.append((data[0], data[1][-1], data[2], data[3]))
        elif data[0] in val_names:
            val_datas.append((data[0], data[1][-1], data[2], data[3]))
        elif data[0] in test_names:
            test_datas.append(data)

    return train_datas, val_datas, test_datas


def task_process(task_func, task_params, save_path, load=True, log=True):
    save_name = os.path.basename(save_path)
    result = None
    if (os.path.exists(save_path)):
        if load:
            result = pickle_load(save_path, log=log)
    else:
        if (log): print(f"Processing {save_name} ...")
        result = task_func(*task_params)
        if (result is None): return
        pickle_dump(result, save_path, log=log)
    return result


def load_data(file_path, split=None, log=True):
    if (log): print(f"loading data {file_path} ...", end=' ')

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if (split):
        lines = [line.strip().split(split) for line in lines]
    else:
        lines = [line.strip() for line in lines]

    if (log): print(f"Success")
    return lines


def pickle_load(file_path, log=True):
    if (log): print(f"loading {file_path} ...", end=' ')

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    if (log): print(f"Success")
    return data


def pickle_dump(data, file_path, log=True):
    if (log): print(f"dumping {file_path} ...", end=' ')

    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

    if (log): print(f"Success")


def json_load(file_path, log=True):
    if (log): print(f"loading json data {file_path} ...", end=' ')

    with open(file_path, 'r', encoding="utf-8") as f:
        json_data = json.load(f)

    if (log): print(f"Success")
    return json_data


def json_dump(data, file_path, indent=4, log=True):
    if (log): print(f"dumping json data {file_path} ...", end=' ')

    with open(file_path, 'w', encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    if (log): print(f"Success")


def auto_makedirs(*dir_paths):
    for dir_path in dir_paths:
        if (not os.path.exists(dir_path)):
            os.makedirs(dir_path)


def add_key(dict_obj, key_name, default_value, append_value=None):
    if key_name not in dict_obj:
        dict_obj[key_name] = default_value
    else:
        if append_value is not None:
            dict_obj[key_name].append(append_value)


def get_all_data_files(data_dir):
    data_black_dir = data_dir + 'black/'
    file_paths_black = []
    for root_path, dir_names, file_names in os.walk(data_black_dir):
        file_names = filter(lambda name: os.path.splitext(name)[-1] == '.xlsx', file_names)
        file_paths_black.extend([os.path.join(root_path, file_name) for file_name in file_names])

    data_white_dir = data_dir + 'white/'
    file_paths_white = []
    for root_path, dir_names, file_names in os.walk(data_white_dir):
        file_names = filter(lambda name: os.path.splitext(name)[-1] == '.xlsx', file_names)
        file_paths_white.extend([os.path.join(root_path, file_name) for file_name in file_names])

    file_paths = file_paths_black + file_paths_white

    return file_paths


def get_confusion_matrix(y_pred, y_true, label_num):
    sns.set()
    f, ax = plt.subplots()

    labels = [i for i in range(label_num)]
    C2 = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(C2, annot=True, ax=ax, fmt='.20g')

    ax.set_title('confusion matrix')
    ax.set_xlabel('pre')
    ax.set_ylabel('true')

    plt.show()

    print("over")
