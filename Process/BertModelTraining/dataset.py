import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, is_train_val_test='Test'):
        # store encodings internally 
        self.encodings = encodings
        self.is_train_val_test = is_train_val_test

    def __len__(self):
        # return the number of samples
        # before
        # if self.is_train_val_test == 'Test':
        #     return len(self.encodings['input_ids'])
        # else:
        #     return self.encodings['input_ids'].shape[0]

        # return len(self.encodings['cls_embeddings'])
        return len(self.encodings['input_ids'])

    def __getitem__(self, i):
        # return dictionary of input_ids, attention_mask, and labels for index i 
        return {key: tensor[i] for key, tensor in self.encodings.items()}
