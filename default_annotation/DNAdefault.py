import pickle
import torch
import torch.nn.functional as F

import pandas as pd
import numpy as np

from model import motifDNAbert
from util import util_file

def main(config, choiceid):

    seqs = util_file.load_test_fasta(config.path_data)
    detectType = ['6mA', '5mC', '4mC']

    model = motifDNAbert.BERT().cuda()

    if choiceid == 1:
        model.load_state_dict(torch.load('../pretrain/Muti_RM/trained_model_51seqs.pkl'))
    elif choiceid == 2:
        model.load_state_dict(torch.load('../pretrain/Muti_RM/trained_model_51seqs.pkl'))
    elif choiceid == 3:
        model.load_state_dict(torch.load('../pretrain/Muti_RM/trained_model_51seqs.pkl'))

    problist = []
    attentionlist = []

    for index, seq in enumerate(seqs):
        logits_data_all, motif_all = motif(model, seq, index)
        problist.extend(logits_data_all)
        attentionlist.extend(motif_all)


def motif(model, seq, seqid):
    logits_data_all = []
    motif_all = []

    original_length = len(seq)
    attention = np.zeros(original_length)
    for pos in range(original_length - 41 + 1):
        seqcut = seq[pos: pos + 41]
        logits, representation = model([seqcut])

        pred_prob_all = F.softmax(logits, dim=1)  # 预测概率 [batch_size, class_num]

        pred_prob_sort = torch.max(pred_prob_all, 1)  # 每个样本中预测的最大的概率 [batch_size]
        pred_class = pred_prob_sort[1]  # 每个样本中预测的最大的概率所在的位置（类别） [batch_size]

        if pred_class[0] == 1:
            logits_data_all.append([seqid, pos + 20, pred_prob_all[1]])
        attention = get_attention(representation, 11, 11)
        print(attention)

        attention_scores = np.array(attention).reshape(np.array(attention).shape[0], 1)
        # 将kmer的值映射到原本的词上
        # print(attention_scores)

        real_scores = get_real_score(attention_scores, 6, "mean")
        # print(real_scores)
        # [0.11406162 0.11283347 0.0976507  0.0891733  0.08977495 0.08209212
        #  0.07083524 0.06235376 0.05608991 0.05148937 0.0478408  0.04692864
        #  0.05176129 0.07523871 0.12612516 0.17707104 0.22449467 0.40165303
        #  0.73878154 0.79585557 0.9348764  1.06195553 1.18059767 1.173981
        #  0.94209728 0.87389846 0.69044034 0.5192499  0.35036664 0.1766051
        #  0.06652493 0.0620664  0.06357612 0.05690048 0.05218404 0.05504905
        #  0.05675806 0.04392732 0.03275132 0.03058121 0.03607595]
        # 可以改成top——k个
        attention[real_scores.index(max(real_scores))] = 1

    for i in range(2, len(seq) - 2):
        if attention[i] == 1:
            motif_all.append([seqid, i - 2, i + 2, seq[i - 2, i + 2]])
            attention[i - 2, i + 2] = 0

    return logits_data_all, motif_all

def format_attention(attention):
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                             "output_attentions=True when initializing your model.")
        squeezed.append(layer_attention.squeeze(0))
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)


def get_attention(representation, start, end):
    attention = representation[-1]
    print(len(attention))
    print(attention[0].shape)
    # print(attention[0].squeeze(0).shape) torch.Size([12, 43, 43])

    """
    attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    attn = format_attention(attention)
    # print(attn.shape) torch.Size([12, 12, 43, 43])

    attn_score = []
    for i in range(1, attn.shape[3] - 1):
        # 这里的0只拿cls, becaue use pool out
        attn_score.append(float(attn[start:end + 1, :, 0, i].sum()))

    # print(len(attn_score)) 41
    # [1.449373722076416, 1.700635552406311, 0.3755369186401367, 0.7071642875671387, 0.09015195816755295, 0.03545517101883888, 0.04521455988287926, 0.0765453428030014, 0.1640062928199768, 0.04963753744959831, 0.03696427121758461, 0.019679294899106026, 0.022457897663116455, 0.005492625758051872, 0.006608381401747465, 0.033757537603378296, 0.03775260969996452, 0.06208576634526253, 0.04476575180888176, 0.09322920441627502, 0.12673263251781464, 0.012487317435443401, 0.03718677908182144, 0.019960680976510048, 0.01909538172185421, 0.02972509153187275, 0.028620678931474686, 0.027425797656178474, 0.057874344289302826, 0.03204841911792755, 0.01482231356203556, 0.02564685419201851, 0.010404390282928944, 0.06448142975568771, 0.02806885726749897, 0.12865057587623596, 0.11877980828285217, 0.113426074385643, 0.041186582297086716, 0.3609130084514618, 0.6371581554412842]
    return attn_score

def get_real_score(attention_scores, kmer, metric):
    # 将kmer的值映射到原本的词上
    # 先创建两个容器放真实的数值
    counts = np.zeros([len(attention_scores) + kmer - 1])
    real_scores = np.zeros([len(attention_scores) + kmer - 1])

    if metric == "mean":
        for i, score in enumerate(attention_scores):
            for j in range(kmer):
                # count 用来数出现的次数，然后平均方法即可
                counts[i + j] += 1.0
                real_scores[i + j] += score

        real_scores = real_scores / counts
    else:
        pass

    return real_scores

if __name__ == '__main__':
    main(["AACGAAGATGAGGACGACTGCGAAAACGATGCATTGCAGGC"],0)