import itertools

from torch import nn
import torch
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier


class Dense(nn.Module):
    def __init__(self, bert_model, dropout=0.5):
        super(Dense, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(768, 12)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 64)
        self.linear5 = nn.Linear(64, 32)
        self.linear6 = nn.Linear(32, 6)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        bert_outputs = self.bert(input_ids=input_id, attention_mask=mask)
        last_hidden_states = bert_outputs['hidden_states'][-1]  # batch * length * dim
        cls_embedding = last_hidden_states[:, 0, :]
        dropout_output = self.dropout(cls_embedding)
        linear_output = self.linear1(dropout_output)
        final_layer = self.relu(linear_output)
        y = self.relu(self.linear2(final_layer))
        y = self.relu(self.linear3(y))
        y = self.relu(self.linear4(y))
        y = self.relu(self.linear5(y))
        y = self.relu(self.linear6(y))
        return y


class MultiDense(nn.Module):
    def __init__(self, bert_model, dropout=0.5, middle_layer_size=64):
        super(MultiDense, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout)
        # self.fc1 = nn.Linear(768, 256)
        # self.fc2 = nn.Linear(256, 64)
        # self.fc3 = nn.Linear(64, 7)
        self.fc1 = nn.Linear(768, middle_layer_size)
        self.fc2 = nn.Linear(middle_layer_size, 7)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, input_id, mask):
        bert_outputs = self.bert(input_ids=input_id, attention_mask=mask)
        last_hidden_states = bert_outputs['hidden_states'][-1]  # batch * length * dim
        print("last_hidden_states:", last_hidden_states.shape)
        cls_embedding = last_hidden_states[:, 0, :]

        x = self.tanh(self.fc1(cls_embedding))
        x = self.dropout(x)
        final_layer = self.fc2(x)
        return final_layer


# raw
class CNN_Dense(nn.Module):
    def __init__(self, bert_model, num_labels, dropout=0.5):
        super(CNN_Dense, self).__init__()
        self.bert = bert_model
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=64, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc = nn.Linear(64, num_labels)

    def forward(self, input_id, mask):
        bert_outputs = self.bert(input_ids=input_id, attention_mask=mask)
        last_hidden_states = bert_outputs['hidden_states'][-1]  # batch * length * dim
        cls_embedding = last_hidden_states[:, 0, :]

        cls_embedding = cls_embedding.unsqueeze(2)
        x = self.conv1(cls_embedding)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def update_dict_return_delitem(dict):
    new_dict = {}
    for pair, dist in dict.items():
        # 删除 pair
        if pair[0] == 'null':
            continue
        else:
            del_index = pair[0]
        break

    # 更新字典
    for pair, dist in dict.items():
        if pair[0] == del_index and pair[1] == del_index:
            continue
        elif pair[0] == del_index:
            new_dict[('null', pair[1])] = dist
        elif pair[1] == del_index:
            new_dict[(pair[0]), 'null'] = dist
        else:
            new_dict[pair] = dist

    return del_index, new_dict


class TextCNN(nn.Module):
    def __init__(self, bert_model, dropout=0.5, num_filters=128, filter_sizes=[2, 3, 4]):
        super(TextCNN, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (size, 768))
            for size in filter_sizes
        ])
        self.fc = nn.Linear(num_filters * len(filter_sizes), 12)

    def forward(self, input_id, mask):
        bert_outputs = self.bert(input_ids=input_id, attention_mask=mask)
        last_hidden_states = bert_outputs['hidden_states'][-1]  # batch * length * dim
        dropout_output = self.dropout(last_hidden_states)  # batch * length * dim
        # TextCNN
        x = dropout_output.unsqueeze(1)
        x = [self.dropout(F.relu(conv(x))).squeeze(3) for conv in self.convs]  # Apply convolutions
        x = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in x]  # Max pool over time
        x = torch.cat(x, 1)
        x = self.fc(x)
        return x


class BiLSTM_Attention(nn.Module):

    def __init__(self, bert_model):
        super(BiLSTM_Attention, self).__init__()

        self.bert = bert_model
        self.n_layers = 2
        self.embedding_dim = 768
        self.hidden_dim = 256
        self.output_dim = 7
        self.dropout = 0.1
        self.bilstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.n_layers, bidirectional=True,
                              dropout=self.dropout, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim * 2, self.output_dim)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, input_id, mask):
        bert_outputs = self.bert(input_ids=input_id, attention_mask=mask)
        last_hidden_states = bert_outputs['hidden_states'][-1]  # batch * length * dim
        cls_embedding = last_hidden_states
        output, (final_hidden_state, final_cell_state) = self.bilstm(cls_embedding)
        output = torch.cat([final_hidden_state[-1, :, :], final_hidden_state[-2, :, :]], dim=-1)
        logit = self.fc(output)
        return logit


def knn_model(k):
    return KNeighborsClassifier(n_neighbors=k, metric="minkowski")


def calc_similarity(e1, e2):
    sim = torch.dot(e1, e2) / (torch.linalg.norm(e1) * torch.linalg.norm(e2))
    return sim
