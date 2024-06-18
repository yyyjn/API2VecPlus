"""
@File         :   main.py
@Time         :   2023/06/08 19:13:30
@Author       :   yjn
@Contact      :   yinjunnan1@gmail.com
@Version      :   1.0
@Desc         :   Overview

"""

from Process.DataGeneration import data_generation
from Process.BertModelTraining import model_training
import numpy as np
from itertools import product
import argparse
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
import torch


def main():
    print(torch.__version__)

    # 1. Data generation
    data_path = data_generation.gain_bert_data()

    with open("alchemy.txt", 'a', encoding='utf-8') as f:
        f.write("Double Dense Model: \n\n")

    # Create argument parser
    parser = argparse.ArgumentParser(description='Process some integers.')

    # Add arguments
    parser.add_argument('--lr', type=int, help='learning rate')
    parser.add_argument('--dr', type=int, help='dropout rate')
    parser.add_argument('--mdi', type=int, help='maximum depth of the input')
    parser.add_argument('--mls', type=int, help='maximum length of the input sequence')
    parser.add_argument('--a', type=int, help='parameter a')
    parser.add_argument('--count', type=int, help='execution count')
    # Parse arguments
    args = parser.parse_args()
    args.lr = 1
    args.dr = 1
    args.mdi = 5
    args.mls = 128
    args.a = 0.1
    args.count = 500

    print("args:", args)

    # Print the parsed arguments
    lr = args.lr * 1e-4
    dr = args.dr * 0.1
    mdi = args.mdi
    mls = args.mls
    a = args.a * 0.1
    i = args.count
    b = -1
    print(f"{i}: LR: {lr:.4f} | DR: {dr:.1f} | MDI: {mdi} | MLS: {mls:<3} | alpha: {a:.1f} | beta: {b:.1f} |")
    # 2. Bert Model Pretraining
    max_data_index = mdi
    bert_model = model_training.train_bert_main(max_data_index, data_path)

    # 3. 生成多分类数据
    # data:raw_file_name, apis, label, pid_count
    datas = data_generation.gain_cls_data()
    target_epoch, acc = model_training.train_cls_main(datas, bert_model, i, lr, dr, mls, a, b)
    with open("alchemy.txt", 'a', encoding='utf-8') as output_file:
        output_file.write(
            f"{i}: LR: {lr:.4f} | DR: {dr:.1f} | MDI: {mdi} | MLS: {mls:<3} | alpha: {a:.1f} | beta: {b:.1f} | ==> Acc: {acc} | Epoch: {target_epoch:<2}\n")


if __name__ == '__main__':
    main()
