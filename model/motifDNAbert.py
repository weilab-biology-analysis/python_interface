import torch
import torch.nn as nn

from transformers import BertTokenizer, BertConfig, BertModel

import sys
import os

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained("bert-base-uncased")

'''DNA bert 模型'''
class BERT(nn.Module):
    def __init__(self, ):
        super(BERT,self).__init__()
        self.kmer = 6
        self.pretrainpath = '../pretrain/DNAbert_6mer'
        self.setting = BertConfig.from_pretrained(
            self.pretrainpath,
            num_labels=2,
            finetuning_task="dnaprom",
            cache_dir=None,
            output_attentions=True
        )

        self.tokenizer = BertTokenizer.from_pretrained(self.pretrainpath)
        self.bert = BertModel.from_pretrained(self.pretrainpath, config=self.setting)
        self.classification = nn.Sequential(
            nn.Linear(768, 20),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(20, 2)
        )

    def forward(self, seqs):
        # print(seqs)
        seqs = list(seqs)
        kmer = [[seqs[i][x:x + self.kmer] for x in range(len(seqs[i]) + 1 - self.kmer)] for i in range(len(seqs))]
        # print(kmer)
        kmers = [" ".join(kmer[i]) for i in range(len(kmer))]
        # print(kmers)
        # print(len(kmers))
        token_seq = self.tokenizer(kmers, return_tensors='pt')
        # print(token_seq)
        input_ids, token_type_ids, attention_mask = token_seq['input_ids'], token_seq['token_type_ids'], token_seq[
            'attention_mask']

        representation = self.bert(input_ids.cuda(), token_type_ids.cuda(), attention_mask.cuda())

        output = self.classification(representation["pooler_output"])

        return output, representation
