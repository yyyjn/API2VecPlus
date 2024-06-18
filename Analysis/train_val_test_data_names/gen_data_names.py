"""
@File         :   main.py
@Time         :   2023/07/22 14:19:35
@Author       :   yjn
@Contact      :   yinjunnan1@gmail.com
@Version      :   1.0
@Desc         :   split data and generate list of train/val/test data names
"""
import os
from Process.utils.utils import split_data, pickle_load, pickle_dump, json_dump

ROOT_DIR = './Process/DataGeneration'
INPUT_DIR = os.path.join(ROOT_DIR, 'inputs')
DATA_DIR = './inputs/data'

OUTPUT_DIR = './Analysis/train_val_test_data_names'


def split_data_mean_iter(type2datanames, name2type, target_types):
    good_num = 0
    for root_path, dir_names, file_names in os.walk(DATA_DIR):
        file_names = filter(lambda name: os.path.splitext(name)[-1] == '.xlsx', file_names)
        for file_name in file_names:
            raw_file_name, _ = os.path.splitext(file_name)
            if raw_file_name in name2type and name2type[raw_file_name] in target_types:
                type_name = name2type[raw_file_name]
                type2datanames[type_name].append(raw_file_name)

            # 引入良性软件类别: V2.0
            if raw_file_name not in name2type:
                if good_num <= 5000:
                    type_name = 'goodware'
                    type2datanames[type_name].append(raw_file_name)
                    good_num += 1

    train_data_names, val_data_names, test_data_names = [], [], []
    for type_name, data_names in type2datanames.items():
        train_data_names_iter, last_data_names = split_data(data_names, split_ratio=0.8, seed=7)
        val_data_names_iter, test_data_names_iter = split_data(last_data_names, split_ratio=0.5, seed=7)

        train_data_names.extend(train_data_names_iter)
        val_data_names.extend(val_data_names_iter)
        test_data_names.extend(test_data_names_iter)

    print("train_data_names len:", len(train_data_names))
    print("val_data_names len:", len(val_data_names))
    print("test_data_names len:", len(test_data_names))
    print("sum:", len(train_data_names) + len(val_data_names) + len(test_data_names))
    return [train_data_names, val_data_names, test_data_names]


def gen_data_names_main():
    type2names = pickle_load(os.path.join(INPUT_DIR, 'type_names_family.pkl'))  # malware_name > family_name
    name2type = {pe_name: type_name for type_name, pe_names in type2names.items() for pe_name in pe_names}
    # raw
    target_types = [type_name for type_name, pe_names in type2names.items() if
                    len(pe_names) >= 200 and type_name not in ('UNKNOW')]

    type2idx = {type_name: i for i, type_name in enumerate(target_types)}

    type2idx['goodware'] = len(type2idx)  # V2.0

    print(f"Class Count: {len(type2idx)}")
    print(f"Classes: {list(type2idx.keys())}")
    print("Classes type2idx:", type2idx)

    type2datanames = {type_name: [] for type_name in type2idx}

    # raw
    data_names = split_data_mean_iter(type2datanames, name2type, target_types)
    # year
    # data_names = split_data_mean_iter_year(type2datanames, name2type, target_types)
    # # test:unkonw
    # data_names = split_data_mean_iter_unknow(type2datanames, name2type, target_types)
    # # test:attack
    # data_names = split_data_mean_iter_attack(type2datanames, name2type, target_types)

    output_path = os.path.join(OUTPUT_DIR, 'data_names.pkl')
    pickle_dump(data_names, output_path)
    output_path = os.path.join(OUTPUT_DIR, 'data_names.json')
    json_dump(data_names, output_path)

    name2typeidx = {name: type2idx[type_name] for type_name, names in type2datanames.items() for name in names}
    output_path = os.path.join(OUTPUT_DIR, 'name2typeidx.pkl')
    pickle_dump(name2typeidx, output_path)
    output_path = os.path.join(OUTPUT_DIR, 'name2typeidx.json')
    json_dump(name2typeidx, output_path)
