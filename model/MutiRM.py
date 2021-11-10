import torch
import numpy as np
from torch import nn
import pickle

class BahdanauAttention(nn.Module):
    """
    input: from RNN module h_1, ... , h_n (batch_size, seq_len, units*num_directions),
                                    h_n: (num_directions, batch_size, units)
    return: (batch_size, num_task, units)
    """
    def __init__(self,in_features, hidden_units,num_task):
        super(BahdanauAttention,self).__init__()
        self.W1 = nn.Linear(in_features=in_features,out_features=hidden_units)
        self.W2 = nn.Linear(in_features=in_features,out_features=hidden_units)
        self.V = nn.Linear(in_features=hidden_units, out_features=num_task)

    def forward(self, hidden_states, values):
        hidden_with_time_axis = torch.unsqueeze(hidden_states,dim=1)

        score = self.V(nn.Tanh()(self.W1(values)+self.W2(hidden_with_time_axis)))
        attention_weights = nn.Softmax(dim=1)(score)
        values = torch.transpose(values,1,2)   # transpose to make it suitable for matrix multiplication
        # print(attention_weights.shape,values.shape)
        context_vector = torch.matmul(values,attention_weights)
        context_vector = torch.transpose(context_vector,1,2)
        return context_vector, attention_weights

class EmbeddingSeq(nn.Module):
    def __init__(self,weight_dict_path):
        """
        Inputs:
            weight_dict_path: path of pre-trained embeddings of RNA/dictionary
        """
        super(EmbeddingSeq,self).__init__()
        weight_dict = pickle.load(open(weight_dict_path,'rb'))

        weights = torch.FloatTensor(list(weight_dict.values())).cuda()
        num_embeddings = len(list(weight_dict.keys()))
        embedding_dim = 300

        self.embedding = nn.Embedding(num_embeddings=num_embeddings,embedding_dim=embedding_dim)
        self.embedding.weight = nn.Parameter(weights)
        self.embedding.weight.requires_grad = False

    def forward(self,x):

        out = self.embedding(x.type(torch.cuda.LongTensor))

        return out

class model_v3(nn.Module):

    def __init__(self,num_task,use_embedding):
        super(model_v3,self).__init__()

        self.num_task = num_task
        self.use_embedding = use_embedding
        if self.use_embedding:
            self.embed = EmbeddingSeq('../pretrain/Muti_RM/embeddings_12RM.pkl') # Word2Vec
            # self.embed = EmbeddingHmm(t=3,out_dims=300) # hmm
            self.NaiveBiLSTM = nn.LSTM(input_size=300,hidden_size=256,batch_first=True,bidirectional=True)
        else:
            self.NaiveBiLSTM = nn.LSTM(input_size=4,hidden_size=256,batch_first=True,bidirectional=True)

        self.Attention = BahdanauAttention(in_features=512,hidden_units=100,num_task=num_task)
        for i in range(num_task):
            setattr(self, "NaiveFC%d" %i, nn.Sequential(
                                       nn.Linear(in_features=512,out_features=128),
                                       nn.ReLU(),
                                       nn.Dropout(),
                                       nn.Linear(in_features=128,out_features=1),
                                       nn.Sigmoid()
                                                    ))

    def forward(self,x):

        if self.use_embedding:
            x = self.embed(x)
        else:
            x = torch.transpose(x,1,2)
        batch_size = x.size()[0]
        # x = torch.transpose(x,1,2)

        output,(h_n,c_n) = self.NaiveBiLSTM(x)
        h_n = h_n.view(batch_size,output.size()[-1])
        context_vector,attention_weights = self.Attention(h_n,output)
        # print(attention_weights.shape)
        outs = []
        for i in range(self.num_task):
            FClayer = getattr(self, "NaiveFC%d" %i)
            y = FClayer(context_vector[:,i,:])
            y = torch.squeeze(y, dim=-1)
            outs.append(y)

        return outs