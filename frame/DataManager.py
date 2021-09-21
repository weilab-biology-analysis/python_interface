import os
import pickle
import torch
import torch.utils.data as Data

import random
import numpy as np
from util import util_file
from random import shuffle


class DataManager():
    def __init__(self, learner):
        self.learner = learner
        self.IOManager = learner.IOManager
        self.visualizer = learner.visualizer
        self.config = learner.config

        self.mode = self.config.mode

        if self.config.cuda:
            self.device = torch.device('cuda')
            torch.cuda.set_device(self.config.device)
            # if self.config.seed:
            #     torch.cuda.manual_seed(self.config.seed)
        else:
            self.device = torch.device('cpu')

        # label:
        self.train_label = None
        self.valid_label = None
        self.test_label = None
        # raw_data: ['MNH', 'APD', ...]
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        # iterator
        self.train_dataloader = None
        self.valid_dataloader = None
        self.test_dataloader = None

    def load_data(self):
        self.train_dataset, self.train_label, self.test_dataset,self.test_label = util_file.load_fasta(self.config.path_data)

        self.train_dataloader = self.construct_dataset(self.train_dataset,self.train_label, self.config.cuda, self.config.batch_size)
        self.test_dataloader = self.construct_dataset(self.test_dataset,self.test_label, self.config.cuda, self.config.batch_size)

        # set max length for model initialization
        # print('Final Max Length: {} (config.max_len: {}, data_max_len:{})'.format(
        #     max(self.config.max_len, self.data_max_len), self.config.max_len, self.data_max_len))
        # if self.config.max_len < self.data_max_len:
        #     self.config.max_len = self.data_max_len

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

    def get_dataloder(self, name):
        return_data = None
        if name == 'train_set':
            return_data = self.train_dataloader
        elif name == 'valid_set':
            return_data = self.valid_dataloader
        elif name == 'test_set':
            return_data = self.test_dataloader

        return return_data


class MyDataSet(Data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
