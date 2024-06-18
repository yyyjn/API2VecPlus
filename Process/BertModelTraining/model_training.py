"""
@File         :   model_training.py
@Time         :   2023/06/14 09:56:42
@Author       :   yjn
@Contact      :   yinjunnan1@gmail.com
@Version      :   1.0
@Desc         :   进行 Bert 模型的预训练
"""
import pickle
from pathlib import Path
import os
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaTokenizer, RobertaConfig, RobertaForMaskedLM, RobertaForSequenceClassification
from transformers import AdamW, pipeline
import torch
from torch import nn
from torch.nn import DataParallel
from tqdm.auto import tqdm
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt
import itertools

from Process.utils.utils import auto_makedirs, pickle_load, split_data, pickle_dump, split_data_by_name_raw, \
    split_data_by_name2, split_data_by_name_concate, split_data_by_name_remove, get_confusion_matrix
from Process.BertModelTraining.dataset import Dataset
from Process.BertModelTraining.cls_models import Dense, TextCNN, MultiDense, CNN_Dense, BiLSTM_Attention, \
    calc_similarity, update_dict_return_delitem
from Process.BertModelTraining.FocalLoss import FocalLoss

ROOT_DIR = './Process/BertModelTraining'
OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs')
MODEL_NAME = 'APIBert'
TOKENIZER_NAME = 'APITokenizer'
BERT_MODEL_DIR = os.path.join(OUTPUT_DIR, MODEL_NAME)
CLS_MODEL_DIR = os.path.join(OUTPUT_DIR, 'BertCls')
TOKENIZER_DIR = os.path.join(OUTPUT_DIR, TOKENIZER_NAME)
CLS_DATALOADER_PATH = os.path.join(OUTPUT_DIR, 'cls_dataloader.pkl')
DATA_NAMES = './Analysis/train_val_test_data_names/data_names.pkl'

auto_makedirs(OUTPUT_DIR, BERT_MODEL_DIR, TOKENIZER_DIR, CLS_MODEL_DIR)

MAX_LENGTH = 512
BATCH_SIZE = 64
BERT_EPOCHS = 2
CLS_EPOCHS = 50
DEVICE_IDS = [0, 5, 6, 8, 9]


def train_tokenizer(paths, save_dir):
    if not os.path.exists(os.path.join(save_dir, 'vocab.json')) and paths is not None:
        print(f"Training tokenizer [{save_dir}]...")

        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(files=paths, special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])

        tokenizer.save_model(save_dir)
    print(f"Loading tokenizer [{save_dir}]...")
    tokenizer = RobertaTokenizer.from_pretrained(save_dir, max_len=MAX_LENGTH)
    print(f"Accept: Loading tokenizer [{save_dir}]...")

    return tokenizer


def build_bert_dataloader(datas, tokenizer, batch_size, save_dir=None):
    print("Building dataloader...")
    encodings = tokenizer.batch_encode_plus(datas, max_length=MAX_LENGTH, pad_to_max_length=True)
    labels = torch.tensor(encodings['input_ids'])
    mask = torch.tensor(encodings['attention_mask'])
    input_ids = labels.detach().clone()

    rand = torch.rand(input_ids.shape)
    mask_arr = (rand < .15) * (input_ids != 0) * (input_ids != 1) * (input_ids != 2)
    for i in range(input_ids.shape[0]):
        # get indices of mask positions from mask array 
        selection = torch.flatten(mask_arr[i].nonzero()).tolist()
        # mask input_ids 
        input_ids[i, selection] = 3  # our custom [MASK] token == 3
    encodings = {'input_ids': input_ids, 'attention_mask': mask, 'labels': labels}

    dataset = Dataset(encodings)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


# raw
def build_cls_dataloader_raw(datas, tokenizer, batch_size, save_dir=None):
    # save_path = os.path.join(save_dir, 'dataloader.pkl')
    # if not os.path.exists(save_path):
    print("Building dataloader...")

    api_seqs = [' '.join(api_seq) for raw_name, api_seq, type_idx, pid_count in datas]
    encodings = tokenizer.batch_encode_plus(api_seqs, max_length=MAX_LENGTH, padding='longest', truncation=True)
    input_ids = torch.tensor(encodings['input_ids'])
    mask = torch.tensor(encodings['attention_mask'])

    names = [raw_name for raw_name, api_seq, type_idx, pid_count in datas]
    type_idxes = [type_idx for raw_name, api_seq, type_idx, pid_count in datas]
    pid_counts = [pid_count for raw_name, api_seq, type_idx, pid_count in datas]

    encodings = {'input_ids': input_ids, 'attention_mask': mask, 'labels': type_idxes, 'names': names,
                 'pid_counts': pid_counts}

    dataset = Dataset(encodings)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # dataiter = iter(loader)
    # sample_data = next(dataiter, None)
    # print(sample_data)

    return loader


def train_bert(model, dataloader, save_dir, device):
    # model = DataParallel(model, device_ids=DEVICE_IDS)
    model.to(device)
    model.train()
    optim = AdamW(model.parameters(), lr=1e-4)

    epochs = BERT_EPOCHS
    for epoch in range(epochs):
        loop = tqdm(dataloader, leave=True, desc=f'Epoch {epoch}')
        for batch in loop:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs.loss
            loss.backward()
            # loss.sum().backward()
            # loss.mean().backward()

            optim.step()
            loop.set_postfix(loss=loss.sum().item())

            input_ids.cpu()
            attention_mask.cpu()
            labels.cpu()
            torch.cuda.empty_cache()

    print(f"Saving Model: {save_dir}")
    auto_makedirs(save_dir)
    model.save_pretrained(save_dir)


def init_bert_model(config, max_data_index):
    # 1. 是否存在目标模型, 存在则直接加载
    target_model_save_dir = os.path.join(BERT_MODEL_DIR, str(max_data_index))
    if os.path.exists(target_model_save_dir):
        print(f"Loading Model: {target_model_save_dir}")
        return RobertaForMaskedLM.from_pretrained(target_model_save_dir)

    # 2. 是否存在已训练的模型, 如果不存在则初始化模型并返回, 存在则找到最新 model 返回
    basic_model_save_dir = os.path.join(BERT_MODEL_DIR, str(1))
    if not os.path.exists(basic_model_save_dir):
        print(f"Initing Model...")
        return RobertaForMaskedLM(config)

    # 3. 寻找最新的 model 继续训练
    for data_index in range(2, max_data_index + 1):
        target_model_save_dir = os.path.join(BERT_MODEL_DIR, str(data_index))
        if not os.path.exists(target_model_save_dir):
            break

    target_model_save_dir = os.path.join(BERT_MODEL_DIR, str(data_index - 1))
    print(f"Loading Model: {target_model_save_dir}")
    return RobertaForMaskedLM.from_pretrained(target_model_save_dir)


def train_bert_main(max_data_index, data_dir):
    # 1. 训练/加载分词器
    dir_path = os.path.join(data_dir, '1')
    paths = [str(x) for x in Path(dir_path).glob('*.txt')]
    tokenizer = train_tokenizer(paths, TOKENIZER_DIR)

    # 2. 初始化模型
    config = RobertaConfig(
        max_position_embeddings=MAX_LENGTH + 2,  # 最大位置嵌入数，即序列最大长度
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,  # 定义了可接受的句子 ID 数量
        vocab_size=tokenizer.vocab_size,  # we align this to the tokenizer vocab_size
    )
    model = init_bert_model(config, max_data_index)
    model.resize_token_embeddings(len(tokenizer))
    # 3. 训练/增量训练
    device = torch.device(f'cuda:{DEVICE_IDS[0]}') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    for data_index in range(1, max_data_index + 1):
        model_save_dir = os.path.join(BERT_MODEL_DIR, str(data_index))
        if os.path.exists(model_save_dir):  # 该数据已经训练过了, 跳过即可
            continue

        dir_path = os.path.join(data_dir, str(data_index))
        paths = [str(x) for x in Path(dir_path).glob('*.pkl')]
        for path in paths:
            datas = pickle_load(path)
            dataloader = build_bert_dataloader(datas, tokenizer, batch_size=BATCH_SIZE)

            train_bert(model, dataloader, model_save_dir, device)

    return model

    fill = pipeline('fill-mask', model=model, tokenizer=tokenizer)

    print("====" * 8)
    # RegQueryValueExW
    result = fill(
        f'CreateProcessInternalW CreateFileW ReadFile CreateFileW LoadLibraryA LoadLibraryA LoadLibraryA CreateFileW WriteFile OpenMutexW RegOpenKeyExW RegQueryValueExW LoadLibraryA CreateProcessInternalW ExitProcess LoadLibraryA LoadLibraryA CreateFileW WriteFile OpenMutexW RegOpenKeyExW {fill.tokenizer.mask_token} LoadLibraryA CreateProcessInternalW ExitProcess LoadLibraryA RegOpenKeyExW RegQueryValueExW ExitProcess')
    for i, item in enumerate(result):
        print(f'{i:<1}: {item["token_str"]}')
    print("====" * 8)
    # ReadFile
    result = fill(
        f'LoadLibraryA LoadLibraryA CreateFileW {fill.tokenizer.mask_token} CreateFileW LoadLibraryA LoadLibraryA CreateFileW ReadFile CreateFileW WriteFile OpenMutexW RegOpenKeyExW RegQueryValueExW LoadLibraryA CreateProcessInternalW RegQueryValueExW LoadLibraryA RegOpenKeyExW RegQueryValueExW')
    for i, item in enumerate(result):
        print(f'{i:<1}: {item["token_str"]}')
    print("====" * 8)

    # OpenMutexW
    result = fill(
        f'CreateFileW WriteFile {fill.tokenizer.mask_token} RegOpenKeyExW LoadLibraryA CreateProcessInternalW RegQueryValueExW ExitProcess')
    for i, item in enumerate(result):
        print(f'{i:<1}: {item["token_str"]}')
    print("====" * 8)


def init_cls_model(bert_model, dr, mls):
    # print("-------------Dense-------------")
    # return Dense(bert_model)
    print("-------------TextCNN-------------")
    return TextCNN(bert_model)
    # print("-------------MultiDense-------------")
    # return MultiDense(bert_model, dropout=dr, middle_layer_size=mls)
    # print("-------------CNNDense-------------")
    # return CNN_Dense(bert_model, num_labels=6)
    # print("-------------BiLSTM_Attention-------------")
    # return BiLSTM_Attention(bert_model)


def train_cls(model, train_dataloader, val_dataloader, test_dataloader, model_save_path, device, lr, a, b):
    # model = DataParallel(model, device_ids=DEVICE_IDS)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    optim = AdamW(model.parameters(), lr=lr)
    # optim = torch.optim.Adam(model.parameters())
    criterion1 = nn.CrossEntropyLoss()
    num_labels = 12
    criterion2 = FocalLoss(num_labels, )
    b = 1.0 - a
    epochs = CLS_EPOCHS
    min_loss = 1e7
    target_epoch = -1
    train_loss_lst, val_loss_lst, train_acc_lst, val_acc_lst = [], [], [], []
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        loop = tqdm(train_dataloader, leave=True, desc=f'Train Epoch {epoch}')
        # by ycc
        # loop = tqdm(test_dataloader, leave=True, desc=f'Train Epoch {epoch}')

        batch_count_train, total_count_train, total_loss_train, total_acc_train = 0, 0, 0, 0

        for batch in loop:
            batch_count_train += 1
            # 清理梯度准备数据
            optim.zero_grad()
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['labels'].to(device)

            total_count_train += len(labels)
            outputs = model(input_ids, attention_mask)
            #
            # 计算指标 loss && acc
            batch_loss = a * criterion1(outputs, labels) + b * criterion2(outputs, labels)

            # l2_lambda = torch.tensor(0.01).to(device)
            # l2_regularization = torch.tensor(0.).to(device)
            # for param in model.parameters():
            #     l2_regularization += torch.norm(param, 2)
            # batch_loss += l2_lambda * l2_regularization

            total_loss_train += batch_loss.item()

            acc_count = (outputs.argmax(dim=1) == labels).sum().item()
            batch_acc = acc_count / len(labels)
            total_acc_train += acc_count
            loop.set_postfix(loss=batch_loss.item(), acc=f'{batch_acc:.2%}')
            # 反向传播
            batch_loss.backward()
            optim.step()
            # 释放数据
            input_ids.cpu()
            attention_mask.cpu()
            labels.cpu()
            torch.cuda.empty_cache()

        # Val
        model.eval()
        loop = tqdm(val_dataloader, leave=True, desc=f'Val Epoch {epoch}')
        batch_count_val, total_count_val, total_loss_val, total_acc_val = 0, 0, 0, 0
        with torch.no_grad():
            for batch in loop:
                batch_count_val += 1
                # 准备数据
                input_ids = batch['input_ids'].cuda()
                attention_mask = batch['attention_mask'].cuda()
                labels = batch['labels'].to(device)

                total_count_val += len(labels)
                outputs = model(input_ids, attention_mask)
                # 计算指标 loss && acc
                batch_loss = a * criterion1(outputs, labels) + b * criterion2(outputs, labels)
                # batch_loss = a * criterion1(outputs, labels.long())
                total_loss_val += batch_loss.item()

                acc_count = (outputs.argmax(dim=1) == labels).sum().item()
                batch_acc = acc_count / len(labels)
                total_acc_val += acc_count
                loop.set_postfix(loss=batch_loss.item(), acc=f'{batch_acc:.2%}')

                # 释放数据
                input_ids.cpu()
                attention_mask.cpu()
                labels.cpu()
                torch.cuda.empty_cache()

        if total_loss_val < min_loss:
            min_loss = total_loss_val
            target_epoch = epoch
            torch.save(model, model_save_path)

        train_loss, val_loss = total_loss_train / batch_count_train, total_loss_val / batch_count_val
        train_acc, val_acc = total_acc_train / total_count_train, total_acc_val / total_count_val
        print(f'''Epochs: {epoch}
        | Train Loss: {total_loss_train / batch_count_train: .4f}
        | Train Accuracy: {total_acc_train / total_count_train: .2%}
        | Val Loss: {total_loss_val / batch_count_val: .4f}
        | Val Accuracy: {total_acc_val / total_count_val: .2%}''')
        train_loss_lst.append(train_loss)
        val_loss_lst.append(val_loss)
        train_acc_lst.append(train_acc)
        val_acc_lst.append(val_acc)

    plt.plot(train_loss_lst, label='Train Loss')
    plt.plot(train_acc_lst, label='Train Accuracy')
    plt.plot(val_loss_lst, label='Val Loss')
    plt.plot(val_acc_lst, label='Val Accuracy')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')

    fig_save_path = os.path.join(CLS_MODEL_DIR, f'loss_acc_{target_epoch}.png')
    plt.savefig(fig_save_path)
    print(f'图表已保存到 {fig_save_path} 文件中。')

    return target_epoch

def test_cls_raw(model, test_dataloader, device):
    # model = DataParallel(model, device_ids=DEVICE_IDS)
    model.to(device)
    model.eval()
    loop = tqdm(test_dataloader, leave=True, desc=f'Test')
    batch_count_test, total_count_test, total_acc_test = 0, 0, 0
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for batch in loop:
            batch_count_test += 1
            # 准备数据
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            true_label = [lbl.tolist() for lbl in labels]
            true_labels.extend(true_label)
            # print(true_labels)
            # input()
            labels = labels.to(device)

            total_count_test += len(labels)
            outputs = model(input_ids, attention_mask)
            pre_ = outputs.argmax(dim=1)
            pred_labels.extend(pre_.tolist())

            # 计算指标 loss && acc
            acc_count = (outputs.argmax(dim=1) == labels).sum().item()
            batch_acc = acc_count / len(labels)
            total_acc_test += acc_count
            loop.set_postfix(acc=f'{batch_acc:.2%}')

            # 释放数据
            input_ids.cpu()
            attention_mask.cpu()
            labels.cpu()
            torch.cuda.empty_cache()

    acc = total_acc_test / total_count_test
    print(f'Test | Test Accuracy: {acc: .2%}')

    # get_confusion_matrix(y_pred=pred_labels, y_true=true_labels, label_num=20)

    normal_analysis(true_labels, pred_labels)
    return f'{acc:.2%}'


def train_cls_main(datas, bert_model, index, lr, dr, mls, a, b):
    global CLS_MODEL_DIR
    CLS_MODEL_DIR = os.path.join(OUTPUT_DIR, f'BertCls_{index}')
    auto_makedirs(CLS_MODEL_DIR)

    device = torch.device(f'cuda:{DEVICE_IDS[0]}') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    print("device:", device)
    bert_model.config.output_hidden_states = True

    # 1. 加载分词器
    tokenizer = train_tokenizer(None, TOKENIZER_DIR)

    # 2. 初始化模型
    model = init_cls_model(bert_model, dr, mls)

    if not os.path.exists(CLS_DATALOADER_PATH):
        data_names = pickle_load(DATA_NAMES)
        train_datas, val_datas, test_datas = split_data_by_name_raw(datas, data_names)
        print(f'''Train Count: {len(train_datas)}
            Val Count: {len(val_datas)}
            Test Count: {len(test_datas)}''')

        # raw
        train_dataloader = build_cls_dataloader_raw(train_datas, tokenizer, batch_size=BATCH_SIZE)
        val_dataloader = build_cls_dataloader_raw(val_datas, tokenizer, batch_size=BATCH_SIZE)
        test_dataloader = build_cls_dataloader_raw(test_datas, tokenizer, batch_size=BATCH_SIZE)
        pickle_dump((train_dataloader, val_dataloader, test_dataloader), CLS_DATALOADER_PATH)

    else:
        (train_dataloader, val_dataloader, test_dataloader) = pickle_load(CLS_DATALOADER_PATH)

    # 3. 训练/增量训练
    model_save_path = os.path.join(CLS_MODEL_DIR, 'best_model.pth')
    target_epoch = 0

    if not os.path.exists(model_save_path):
        target_epoch = train_cls(model, train_dataloader, val_dataloader, test_dataloader, model_save_path, device, lr,
                                 a, b)

    print(f"Loading Model: {model_save_path}")
    best_model = torch.load(model_save_path)
    acc = test_cls_raw(best_model, test_dataloader, device)

    return target_epoch, acc


from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report

def normal_analysis(y_trues, y_preds):
    # tn, fp, fn, tp = confusion_matrix(y_trues, y_preds).ravel()
    micro_precision = precision_score(y_trues, y_preds, average="micro")
    micro_recall = recall_score(y_trues, y_preds, average="micro")
    micro_f1 = f1_score(y_trues, y_preds, average="micro")

    macro_precision = precision_score(y_trues, y_preds, average="macro")
    macro_recall = recall_score(y_trues, y_preds, average="macro")
    macro_f1 = f1_score(y_trues, y_preds, average="macro")

    weighted_precision = precision_score(y_trues, y_preds, average="weighted")
    weighted_recall = recall_score(y_trues, y_preds, average="weighted")
    weighted_f1 = f1_score(y_trues, y_preds, average="weighted")

    result = f'Accuracy_score: {accuracy_score(y_trues, y_preds):.4f}\n'
    result += f'Micro precision: {micro_precision:.4f}\n'
    result += f'Micro recall: {micro_recall:.4f}\n'
    result += f'Micro score: {micro_f1:.4f}\n'

    result += f'Macro precision: {macro_precision:.4f}\n'
    result += f'Macro recall: {macro_recall:.4f}\n'
    result += f'Macro score: {macro_f1:.4f}\n'

    result += f'Weighted precision: {weighted_precision:.4f}\n'
    result += f'Weighted recall: {weighted_recall:.4f}\n'
    result += f'Weighted score: {weighted_f1:.4f}\n'
    print(result)
