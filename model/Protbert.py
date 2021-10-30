import torch
import torch.nn as nn

from transformers import BertTokenizer, BertModel

import sys
import os
import re

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained("bert-base-uncased")

'''Prot bert 模型'''
class BERT(nn.Module):
    def __init__(self, config):
        super(BERT,self).__init__()
        self.config = config

        # 加载预训练模型参数
        self.model_name = config.model
        if self.model_name == 'prot_bert_bfd':
            self.pretrainpath = '../pretrain/prot_bert_bfd'
        elif self.model_name == 'prot_bert':
            self.pretrainpath = '../pretrain/prot_bert'

        self.tokenizer = BertTokenizer.from_pretrained(self.pretrainpath)
        self.bert = BertModel.from_pretrained(self.pretrainpath)
        self.classification = nn.Sequential(
            nn.Linear(768, 20),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(20, 2)
        )

    def forward(self, seqs):
        sequences = re.sub(r"[UZOB]", "X", seqs)
        token_seq = self.tokenizer(sequences, return_tensors='pt')
        # print(token_seq)
        input_ids, token_type_ids, attention_mask = token_seq['input_ids'], token_seq['token_type_ids'], token_seq[
            'attention_mask']
        if self.config.cuda:
            representation = self.bert(input_ids.cuda(), token_type_ids.cuda(), attention_mask.cuda())["pooler_output"]
        else:
            representation = self.bert(input_ids, token_type_ids, attention_mask)["pooler_output"]

        output = self.classification(representation)

        return output, representation
