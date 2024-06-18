"""
@File         :   main.py
@Time         :   2023/06/08 16:32:44
@Author       :   yjn
@Contact      :   yinjunnan1@gmail.com
@Version      :   1.0 > 2.0
@Desc         :   Given API sequences we tend to generate pretraining data 
                    for the Bert model using the graph model and random walk 
                    algorithm in the API2Vec model.
                  malware families + goodware multi-classification
"""
import os
from Process.utils.utils import auto_makedirs, pickle_dump, task_process, json_dump, pickle_load
from Process.utils.data_process_utils import load_tpg_tag_datas_without_label, load_raw_sequences_by_types
from Process.utils.random_walk_algs import random_walk_basic
import numpy as np

DATA_DIR = './inputs/data'
ROOT_DIR = './Process/DataGeneration'
INPUT_DIR = os.path.join(ROOT_DIR, 'inputs')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs')

SAVE_TPG_TAG_DATAS_PATH = os.path.join(OUTPUT_DIR, 'tpg_tag_datas.pkl')
SAVE_RAW_SEQ_DATAS_PATH = os.path.join(OUTPUT_DIR, 'raw_seqs_data.pkl')

RANDOM_WALK_TYPE = 'random_walk_basic'
SAVE_RANDOM_WALK_DATAS_DIR = os.path.join(OUTPUT_DIR, RANDOM_WALK_TYPE + '_datas')
auto_makedirs(OUTPUT_DIR, SAVE_RANDOM_WALK_DATAS_DIR, )

MULTI_PROCESS_WORKER = 16
RANDOM_WALK_TIMES = 5
RANDOM_WALK_STEPS = 49
BATCH_SIZE = 10_000


def shape_random_walk_datas(random_walk_datas):
    assert len(random_walk_datas) > BATCH_SIZE, "Wrong Random Walk Datas"

    random_walk_datas = [" ".join(paths) for random_walk_data in random_walk_datas for paths in random_walk_data]
    np.random.shuffle(random_walk_datas)

    batch_results, cur_index = [], 1
    while cur_index + BATCH_SIZE <= len(random_walk_datas):  # 不足 10w 的数据舍去了
        batch_results.append(random_walk_datas[cur_index: cur_index + BATCH_SIZE])
        cur_index += BATCH_SIZE
    if cur_index < len(random_walk_datas):
        batch_results.append(random_walk_datas[cur_index:])
    return batch_results


def gain_bert_data():
    target_dir = os.path.join(SAVE_RANDOM_WALK_DATAS_DIR, str(RANDOM_WALK_TIMES))

    if os.path.exists(target_dir):
        return SAVE_RANDOM_WALK_DATAS_DIR
    # 1. 读取 API 序列
    tpg_tag_datas = task_process(load_tpg_tag_datas_without_label,
                                 (DATA_DIR, MULTI_PROCESS_WORKER),
                                 SAVE_TPG_TAG_DATAS_PATH,
                                 load=True,
                                 log=True)

    # 2. 执行随机游走 + 保存游走数据
    for i in range(1, RANDOM_WALK_TIMES + 1):
        random_walk_datas = random_walk_basic(tpg_tag_datas, RANDOM_WALK_STEPS)
        batch_datas = shape_random_walk_datas(random_walk_datas)

        save_dir = os.path.join(SAVE_RANDOM_WALK_DATAS_DIR, str(i))
        if os.path.exists(save_dir):
            continue
        auto_makedirs(save_dir)
        for j, batch_data in enumerate(batch_datas, 1):
            pickle_dump(batch_data, os.path.join(save_dir, f'{j}.pkl'), log=True)
            with open(os.path.join(save_dir, f'{j}.txt'), 'w', encoding='utf-8') as f:
                f.write('\n'.join(batch_data))

    return SAVE_RANDOM_WALK_DATAS_DIR


def gain_cls_data():
    name2pid = pickle_load(os.path.join(INPUT_DIR, 'name2pid.pkl'))  # malware_name > pid_count
    type2names = pickle_load(os.path.join(INPUT_DIR, 'type_names_family.pkl'))  # malware_name > family_name
    name2type = {pe_name: type_name for type_name, pe_names in type2names.items() for pe_name in pe_names}
    # raw
    target_types = [type_name for type_name, pe_names in type2names.items() if
                    len(pe_names) >= 200 and type_name not in ('UNKNOW')]  # 2020 删除sytro ('UNKNOW', 'sytro')
    type2idx = {type_name: i for i, type_name in enumerate(target_types)}

    type2idx['goodware'] = len(type2idx)  # V2.0

    print(f"Class Count: {len(type2idx)}")
    print(f"Classes: {list(type2idx.keys())}")
    print("Classes type2idx:", type2idx)

    # 1. 遍历文件夹获取 api sequences 序列以及类别标签 
    # (name, api_seqs, type_idx, pid_count) * N
    datas = task_process(load_raw_sequences_by_types,
                         (DATA_DIR, name2pid, name2type, target_types, type2idx, MULTI_PROCESS_WORKER),
                         SAVE_RAW_SEQ_DATAS_PATH,
                         load=True,
                         log=True)

    return datas