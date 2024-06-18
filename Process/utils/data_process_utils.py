"""
@File         :   data_process_utils.py
@Author       :   yjn
@Contact      :   yinjunnan1@gmail.com
@Version      :   1.0 > 2.0
"""
import multiprocessing
import pickle

import pandas as pd
from tqdm import tqdm
from Process.utils.utils import add_key, pickle_load
from Process.utils.judge_api_args_property import *
import os

def _load_xlsx_data(file_path):
    data = pd.read_excel(file_path)
    # data = data.sort_values('timestamp')
    apis = data['apicall'].tolist()
    pids = data['pid'].tolist()
    _apis2type = data['category'].tolist()
    apis2type = {}
    for api, api2type in zip(apis, _apis2type):
        apis2type[api] = api2type

    args = data['args'].tolist()
    ret_value = data['return'].tolist()

    return apis, pids, apis2type, args, ret_value


def _load_tpg_tag_datas_without_label_main(file_path):
    apis, pids, apis2type, args, rets = _load_xlsx_data(file_path)

    apis2arg = []  # api + 对应的参数 ,判断api的属性，作为边的属性
    rule = False
    for api, api_args in zip(apis, args):
        ret = []
        api_args_dict_list = eval(api_args)

        if rule:
            for arg_name_value in api_args_dict_list:
                api_name = api
                value = arg_name_value['value']
                if value.find('C:\\') != -1 or value.find('c:\\') != -1 or value.find('D:\\') != -1 or value.find(
                        'd:\\') != -1 or value.find('E:\\') != -1 or value.find('e:\\') != -1:
                    retnum = Rules(api_name, value)
                else:
                    retnum = OtherRules(value)

                ret.append(retnum)

            if 0 in ret:
                apis2arg.append(0)
            elif 0 not in ret and 1 in ret:
                apis2arg.append(1)
            else:
                apis2arg.append(2)

        else:
            apis2arg.append(api_args_dict_list)

    add_ret = True
    if add_ret:

        # arg:ret 每个api的参数，以及其返回值
        for api_arg, ret in zip(apis2arg, rets):
            api_arg.append(ret)

        _api2args2ret = apis2arg

    # 时间索引映射
    pid2apis = {}  # pid 对应的 api 序列
    pid2tidxes = {}  # pid 对应的 时间索引 序列
    api2tidx = {}  # api 对应的最早时间索引
    for t, (api, pid) in enumerate(zip(apis, pids), start=1):
        add_key(pid2apis, pid, [api, ], api)
        add_key(pid2tidxes, pid, [t, ], t)
        add_key(api2tidx, api, t, append_value=None)

    pid2edges2tidxes = {}  # pid(tag) 对应的边集 以及边对应的时间索引序列
    for pid, apis in pid2apis.items():
        # 边
        edges = list(zip(apis, apis[1:]))
        # 边所对应的值
        edges2arg = list(zip(apis2arg, apis2arg[1:]))
        edges2ret = list(zip(rets, rets[1:]))
        # 增加了ret之后的
        apis2arg2ret = list(zip(_api2args2ret, _api2args2ret[1:]))
        tidxes = pid2tidxes[pid][1:]
        assert len(edges) == len(tidxes), "TODO"

        pid2edges2tidxes[pid] = {}
        if rule:
            for edge, tidx, edge_arg in zip(edges, tidxes, edges2arg):
                value_ = {tidx: edge_arg}
                # add_key(pid2edges2tidxes[pid], tuple(edge), [tidx, ], value_)
                add_key(pid2edges2tidxes[pid], tuple(edge), [value_, ], value_)
        else:
            for edge, tidx, edges_args_ret in zip(edges, tidxes, apis2arg2ret):
                value_ = {tidx: edges_args_ret}
                add_key(pid2edges2tidxes[pid], tuple(edge), [value_, ], value_)

    return apis, pid2tidxes, pid2edges2tidxes, api2tidx, apis2type, args


def load_tpg_tag_datas_without_label(data_dir, worker=4):
    # 1. 获取目标文件的路径集合: 遍历数据文件夹 + 按照文件类型过滤 + 拼接路径
    file_paths = []
    for root_path, dir_names, file_names in os.walk(data_dir):
        file_names = filter(lambda name: os.path.splitext(name)[-1] == '.xlsx', file_names)
        file_paths.extend([os.path.join(root_path, file_name) for file_name in file_names])

    # 2. 多进程读取文件获取 tpg 与 tag 的数据：获取 
    # ## pid2tidxes, PID对应时间索引列表：用于 TPG 游走，选择有效时间内的 TPG 节点
    # ## pid2edges2tidxes, PID对应API先后调用边的列表 + 先后调用边对应时间索引列表：用于 Networkx 中图的构建 + 用于边的选择以及当前时间索引的更新
    # ## api2tidx, API对应首次执行时间索引列表：用于初始化最初的当前时间索引
    with multiprocessing.Pool(processes=worker) as p:
        tpg_tag_datas = list(tqdm(
            p.imap(_load_tpg_tag_datas_without_label_main, file_paths),
            total=len(file_paths),
            desc='load_tpg_tag_datas_without_label'))

    return tpg_tag_datas


import random


def attack_insert(api_names, insert_times):
    file_path = './1-resource/old/attack_patterns.pkl'
    with open(file_path, 'rb') as f:
        attack_patterns = pickle.load(f)

    attack_patterns = random.choices(attack_patterns, k=insert_times)

    # for attack_pattern in attack_patterns:
    #     for i in range(len(attack_pattern)):
    #         if attack_pattern[i].endswith('_S') or attack_pattern[i].endswith('_F'):
    #             attack_pattern[i] = attack_pattern[i][:-2]
    #
    # # print(attack_patterns)

    insert_indexs = sorted(random.choices(
        list(range(len(api_names))), k=insert_times))
    insert_indexs.append(len(api_names))

    pieces = []

    pre_index = 0
    for idx in insert_indexs:
        pieces.append(api_names[pre_index:idx])
        pre_index = idx

    assert len(attack_patterns) == len(pieces) - 1, "Wrong split pieces"

    ret_api_names, ret_pids, ret_args = [], [], []
    for piece, attack_pattern in zip(pieces[:-1], attack_patterns):
        api_names_piece = piece

        ret_api_names.extend(api_names_piece)

        attack_api_names_piece = [name[:-2] for name in attack_pattern]

        ret_api_names.extend(attack_api_names_piece)

    ret_api_names.extend(pieces[-1][0])

    return ret_api_names


def _load_raw_sequences_by_types_main(params):
    raw_file_name, file_type_idx, pid_count, file_path = params

    apis, pids, _, _, _ = _load_xlsx_data(file_path)
    while any([item in apis for item in ['ExitWindowsEx', 'DeleteService', 'CreateServiceA', 'CreateServiceA']]):
        for item in ['ExitWindowsEx', 'DeleteService', 'CreateServiceA', 'CreateServiceA']:
            if item in apis:
                apis.remove(item)

    file_path ='./API2Vec/Analysis/train_val_test_data_names/data_names.pkl'
    with open(file_path, 'rb') as f:
        train_data_names, val_data_names, test_data_names = pickle.load(f)
    # attack
    if raw_file_name in test_data_names:
        attack_flag = True
    else:
        attack_flag = False
    if attack_flag:
        length = len(apis)
        insert_times = min(5, max(1, length // 10))
        ret_api_names = attack_insert(apis, insert_times)
        return raw_file_name, ret_api_names, file_type_idx, pid_count
    else:
        return raw_file_name, apis, file_type_idx, pid_count

def load_raw_sequences_by_types(data_dir, name2pid, name2type, target_types, type2idx, worker=4):
    # 1. 获取目标文件的路径集合: 遍历数据文件夹 + 按照文件类型过滤 + 拼接路径
    inputs = []
    # by ycc good_num
    good_num = 0
    unknow_num = 0
    for root_path, dir_names, file_names in os.walk(data_dir):
        file_names = filter(lambda name: os.path.splitext(name)[-1] == '.xlsx', file_names)
        for file_name in file_names:
            raw_file_name, _ = os.path.splitext(file_name)
            # 目标文件应是恶意文件 && 目标文件的家族应当在目标家族内, 否则过滤掉
            # 引入良性软件类别: V2.0
            if raw_file_name in name2type and name2type[raw_file_name] in target_types:
                pid_count = name2pid[raw_file_name]
                file_type = name2type[raw_file_name]
                file_type_idx = type2idx[file_type]
                file_path = os.path.join(root_path, file_name)

                # raw
                inputs.append((raw_file_name, file_type_idx, pid_count, file_path))
                # # 2019
                # inputs.append((raw_file_name, file_type_idx, pid_count, file_path))
                # 2020
                # if file_type != 'sytro':
                #     inputs.append((raw_file_name, file_type_idx, pid_count, file_path))


            # if raw_file_name not in name2type:
            #     pid_count = name2pid[raw_file_name]
            #     file_type = "goodware"
            #     file_type_idx = type2idx[file_type]
            #     file_path = os.path.join(root_path, file_name)
            #     inputs.append((raw_file_name, file_type_idx, pid_count, file_path))

            # by ycc good_num
            if raw_file_name not in name2type:
                if good_num <= 5000:
                    good_num += 1
                    pid_count = name2pid[raw_file_name]
                    file_type = "goodware"
                    file_type_idx = type2idx[file_type]
                    file_path = os.path.join(root_path, file_name)
                    inputs.append((raw_file_name, file_type_idx, pid_count, file_path))

    # 2. 多进程读取文件获取 文件名称序列, 原始 API 序列，类别序列：获取
    with multiprocessing.Pool(processes=worker) as p:
        # (name, api_seqs, type_idx, pid_count) * N
        datas = list(tqdm(
            p.imap(_load_raw_sequences_by_types_main, inputs),
            total=len(inputs),
            desc='_load_raw_sequences_by_types_main'))

    return datas
