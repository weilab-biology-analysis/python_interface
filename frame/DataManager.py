import os
import pickle
import torch
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils
import random
import numpy as np
from util import util_file
from util import util_transGraph

def collate_fn(data):
    # print(data)
    data.sort(key=lambda data: len(data[0]), reverse=True)
    train_data = [tupledata[0] for tupledata in data]
    label_data = [tupledata[1] for tupledata in data]
    # data_length = [len(data) for data in train_data]
    train_data = rnn_utils.pad_sequence(train_data, batch_first=True, padding_value=0)
    # print(train_data)
    # print(label_data)

    return train_data, torch.cuda.LongTensor(label_data)

class DataManager():
    def __init__(self, learner):
        self.learner = learner
        self.IOManager = learner.IOManager
        self.visualizer = learner.visualizer
        self.config = learner.config

        self.mode = self.config.mode

        # label:
        self.train_label = None
        self.test_label = None
        # raw_data: ['MNH', 'APD', ...]
        self.train_dataset = None
        self.test_dataset = None
        # iterator
        self.train_dataloader = None
        self.test_dataloader = None

        self.token2index = None

    def SL_train_load_data(self):
        self.train_dataset, self.train_label, self.test_dataset,self.test_label = util_file.load_fasta(self.config.path_data)

        if self.config.model in ["3mer_DNAbert","4mer_DNAbert","5mer_DNAbert","6mer_DNAbert"]:
            # print("dna_bert_data_process")
            self.train_dataloader = self.construct_dataset(self.train_dataset, self.train_label, self.config.cuda,
                                                           self.config.batch_size)
            self.test_dataloader = self.construct_dataset(self.test_dataset, self.test_label, self.config.cuda,
                                                          self.config.batch_size)
        elif self.config.model in ['TextGCN', '']:
            '''
            the parts process GNN
            '''
            print("graph_data_process")
            Graph = util_transGraph.CreateTextGCNGraph(self.config.path_data)
            self.train_dataloader = Graph
            # adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = Graph.load_corpus()
            # t_support, num_supports = Graph.get_suppport(0)
            # # t_features, t_y_train, t_y_val, t_y_test, t_train_mask, tm_train_mask = \
            # #     Graph.post_process(adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size,
            # #                        test_size)
            # self.train_dataloader = Graph.post_process(adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size,
            #                        test_size)

        else:
            # print("token_data_process")
            if self.config.type == 'DNA':
                self.token2index = pickle.load(open('../data/statistic/DNAtoken2index.pkl', 'rb'))
            elif self.config.type == 'RNA':
                self.token2index = pickle.load(open('../data/statistic/RNAtoken2index.pkl', 'rb'))
            elif self.config.type == 'RNA':
                self.token2index = pickle.load(open('../data/statistic/proteintoken2index.pkl', 'rb'))
            self.train_dataloader = self.construct_dataset_with_same_len(self.train_dataset, self.train_label, self.config.cuda,
                                                           self.config.batch_size)
            self.test_dataloader = self.construct_dataset_with_same_len(self.test_dataset, self.test_label, self.config.cuda,
                                                          self.config.batch_size)

        # set max length for model initialization
        # print('Final Max Length: {} (config.max_len: {}, data_max_len:{})'.format(
        #     max(self.config.max_len, self.data_max_len), self.config.max_len, self.data_max_len))
        # if self.config.max_len < self.data_max_len:
        #     self.config.max_len = self.data_max_len

    def SL_test_load_data(self):
        pass

    def construct_dataset(self, sequences, labels, cuda, batch_size):
        # if cuda:
        #     input_ids, labels = torch.cuda.LongTensor(sequences), torch.cuda.LongTensor(labels)
        # else:
        #     input_ids, labels = torch.LongTensor(sequences), torch.LongTensor(labels)
        if cuda:
            labels = torch.cuda.LongTensor(labels)
        else:
            labels = torch.LongTensor(labels)
        dataset = MyDataSet(sequences, labels)
        data_loader = Data.DataLoader(dataset,
                                      batch_size=batch_size,
                                      shuffle=True)
        return data_loader

    def construct_dataset_with_same_len(self, sequences, labels, cuda, batch_size):
        # if cuda:
        #     input_ids, labels = torch.cuda.LongTensor(sequences), torch.cuda.LongTensor(labels)
        # else:
        #     input_ids, labels = torch.LongTensor(sequences), torch.LongTensor(labels)
        if cuda:
            labels = torch.cuda.LongTensor(labels)
        else:
            labels = torch.LongTensor(labels)
        index_list = []
        max_len = 0
        for seq in sequences:
            seq_index = [self.token2index[token] for token in seq]
            seq_index = [self.token2index['[CLS]']] + seq_index
            # print(self.token2index)
            index_list.append(torch.tensor(seq_index))
            if len(seq) > max_len:
                max_len = len(seq)
        self.config.max_len = max_len + 1
        # print(max_len)
        # data = rnn_utils.pad_sequence(index_list, batch_first=True)
        dataset = MyDataSet(index_list, labels)
        data_loader = Data.DataLoader(dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      collate_fn=collate_fn)
        return data_loader

    def get_dataloder(self, name):
        return_data = None
        if name == 'train_set':
            return_data = self.train_dataloader
        elif name == 'test_set':
            return_data = self.test_dataloader

        return return_data

    # def collate_fn(data):
    #     # print(data)
    #     data.sort(key=lambda data: len(data[0]), reverse=True)
    #     train_data = [tupledata[0] for tupledata in data]
    #     label_data = [tupledata[1] for tupledata in data]
    #     # data_length = [len(data) for data in train_data]
    #     train_data = rnn_utils.pad_sequence(train_data, batch_first=True, padding_value=0)
    #     # print(train_data)
    #     # print(label_data)
    #
    #     return train_data, torch.cuda.LongTensor(label_data)


class MyDataSet(Data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
