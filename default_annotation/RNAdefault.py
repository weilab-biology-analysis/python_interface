import pickle
import torch
import pandas as pd
import numpy as np

from model import MutiRM

def main(seqs):
    RMs = ['Am', 'Cm', 'Gm', 'Um', 'm1A', 'm5C', 'm5U', 'm6A', 'm6Am', 'm7G', 'Psi', 'AtoI']

    num_task = 12
    top = 3
    alpha = 0.1
    att_window = 5 # motif的查找窗口
    verbose = False
    original_length = len(seqs)
    check_pos = original_length - 51 + 1

    model = MutiRM.model_v3(num_task=num_task, use_embedding=True).cuda()
    model.load_state_dict(torch.load('../pretrain/Muti_RM/trained_model_51seqs.pkl'))

    neg_prob = pd.read_csv('../pretrain/Muti_RM/neg_prob.csv', header=None, index_col=0)

    probs = np.zeros((num_task, check_pos))
    p_values = np.zeros((num_task, check_pos))
    labels = np.zeros((num_task, original_length))
    attention = np.zeros((num_task, original_length))

    embeddings_dict = pickle.load(open('../pretrain/Muti_RM/embeddings_12RM.pkl', 'rb'))

    str_out = []
    print('*' * 24 + 'Reporting' + '*' * 24)
    str_out.append('*' * 24 + 'Reporting' + '*' * 24)

    for pos in range(original_length - 51 + 1):
        cutted_seqs = seqs[pos:pos + 51]
        # print(cutted_seqs) "CCTCTGAACCCCCAACACTCTGGCCCATCGGGGTGACGGATATCTGCTTTT"

        seqs_kmers_index = seq2index([cutted_seqs], embeddings_dict)
        # print(seqs_kmers_index) [[2],[12],[11],[13]....]

        seqs_kmers_index = torch.transpose(torch.from_numpy(seqs_kmers_index), 0, 1)

        # Evaluate and cal Attention weights
        attention_weights, y_preds = evaluate(model, seqs_kmers_index)
        # print(y_preds) 12个种类的1个值
        # [tensor([9.1687e-05], device='cuda:0', grad_fn=<SqueezeBackward1>), tensor([0.0969], device='cuda:0', grad_fn=<SqueezeBackward1>), tensor([8.4767e-11], device='cuda:0', grad_fn=<SqueezeBackward1>), tensor([5.7640e-09], device='cuda:0', grad_fn=<SqueezeBackward1>), tensor([0.0002], device='cuda:0', grad_fn=<SqueezeBackward1>), tensor([0.0970], device='cuda:0', grad_fn=<SqueezeBackward1>), tensor([1.2085e-09], device='cuda:0', grad_fn=<SqueezeBackward1>), tensor([7.6775e-15], device='cuda:0', grad_fn=<SqueezeBackward1>), tensor([1.6960e-06], device='cuda:0', grad_fn=<SqueezeBackward1>), tensor([3.6529e-12], device='cuda:0', grad_fn=<SqueezeBackward1>), tensor([1.1736e-08], device='cuda:0', grad_fn=<SqueezeBackward1>), tensor([1.3682e-05], device='cuda:0', grad_fn=<SqueezeBackward1>)]
        total_attention = cal_attention(attention_weights)
        # attention_weights就是49*12的一个向量,每一个种类的weights就是score,3mers将51变成49,49维输进去就是50个score
        # total_attention就是1*12*51的一个向量,就是一些变换

        # print(attention_weights.shape[1])
        # print(total_attention)

        y_prob = [y_pred.detach().cpu().numpy()[0] for y_pred in y_preds]
        # print(y_prob) [9.1686576e-05, 0.096862465, 8.476671e-11, 5.7640124e-09, 0.0001716252, 0.097024314, 1.2085323e-09, 7.677543e-15, 1.6960197e-06, 3.6529174e-12, 1.1735641e-08, 1.3682374e-05]

        for k in range(num_task):
            bool = neg_prob.iloc[k, :] > y_prob[k]

            p_value = np.sum(bool) / len(bool)

            if p_value < alpha:
                labels[k, pos + 25] = 1
            p_values[k, pos] = p_value
            probs[k, pos] = y_prob[k]

        index_list = [i for i, e in enumerate(labels[:, pos + 25]) if e == 1]
        # print(labels)
        print(index_list)
        if index_list == []:
            if verbose:
                sentense = 'There is no modification site at %d ' % (pos + 26)
                print(sentense)
                str_out.append(sentense)
        else:
            for idx in index_list:
                if verbose:
                    sentense = '%s is predicted at %d with p-value %.4f and alpha %.3f' % (
                        RMs[idx], pos + 26, p_values[idx, pos], args.alpha)
                    print(sentense)
                    str_out.append(sentense)

                this_attention = total_attention[0, idx, :]
                # this_attention是当前类别固定，label被选中时的attention值
                position_dict = highest_x(this_attention, w=att_window)
                # print(position_dict) 每三位一组，取出前11位分值最高的组
                # {1: (2.8485403656959534, 24, 26), 2: (0.025789543986320496, 7, 9), 3: (0.02112709730863571, 3, 5), 4: (0.006381317798513919, 11, 13), 5: (0.002240629750303924, 20, 22), 6: (0.0020732737029902637, 28, 30), 7: (0.0016388743533752859, 43, 45), 8: (0.0010588565783109516, 33, 35), 9: (0.0010333763784728944, 39, 41), 10: (0.0006401336431736127, 16, 18), 11: (0.0003925530327251181, 47, 49)}
                edge = pos

                starts = []
                ends = []
                scores = []
                # top决定只取前几个
                for j in range(1, top + 1):
                    score, start, end = position_dict[j]
                    starts.append(start + edge)
                    ends.append(end + edge)
                    scores.append(score)

                    attention[idx, start + edge:end + edge + 1] = 1

def word2index_(my_dict):
    word2index = dict()
    for index, ele in enumerate(list(my_dict.keys())):
        word2index[ele] = index

    return word2index

def mapfun(x,my_dict):
    if x not in list(my_dict.keys()):
        return None
    else:
        return word2index_(my_dict)[x]

def seq2index(seqs,my_dict,window=3,save_data=False):
    """
    Convert single RNA sequences to k-mers representation.
        Inputs: ['ACAUG','CAACC',...] of equal length RNA seqs
        Example: 'ACAUG' ----> [ACA,CAU,AUG] ---->[21,34,31]
    """

    num_samples = len(seqs)
    temp = []
    for k in range(num_samples):
        length = len(seqs[k])
        seqs_kmers = [seqs[k][i:i+window] for i in range(0,length-window+1)]
        temp.append(seqs_kmers)


    seq_kmers = pd.DataFrame(data = np.concatenate(temp,axis=0))

    # load pretained word2vec embeddings

    word2index = word2index_(my_dict)

    seq_kmers_index = seq_kmers.applymap(lambda x: mapfun(x,my_dict))


    return seq_kmers_index.to_numpy()

def evaluate(model, input_x,model_path=None):
    """
    Calculate the attention weights and predicted probabilities
    """
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    model.eval()

    y_pred = model(input_x)
    x = model.embed(input_x)
    output,(h_n,c_n) = model.NaiveBiLSTM(x)
    h_n = h_n.view(-1,output.size()[-1])
    context_vector,attention_weights = model.Attention(h_n,output)

    return attention_weights, y_pred

def cal_attention(total_attention_weights):
    """
    Unwarp the 3-mers inputs attention_weights and sum to single nucleotide
        Inputs: Attention weights shape [batch_size, length, num_class]
        Outputs: Unwarped Attention weights shape [batch_size, num_class, length+2]
    """
    num_class = total_attention_weights.shape[-1] # 12
    length = total_attention_weights.shape[1] + 2 # 3-mers输入要加2
    num_samples = total_attention_weights.shape[0]
    total_attention = np.zeros((num_samples,num_class,length)) # (1*12*50)
    for k in range(num_samples):
        tmp = []
        for i in range(num_class):
            tmp.append(cal_attention_every_class(total_attention_weights[k,:,i].detach().cpu().numpy()))
        tmp = np.concatenate(tmp,axis=0)

        total_attention[k,:] = tmp
    return total_attention

def cal_attention_every_class(attention_weights):
    length = attention_weights.shape[0]
    attention = np.zeros((1,length+2))
    for i in range(length+2):
        # unravel 3-mers attention
        if i == 0:
            attention[:,0] = attention_weights[0]
        elif i == 1:
            attention[:,1] = attention_weights[0] + attention_weights[1]
        elif i == length +1:
            attention[:,i] = attention_weights[i-2]
        elif i == length:
            attention[:,i] = attention_weights[i-2] + attention_weights[i-1]
        else:
            attention[:,i] = attention_weights[i-2]+attention_weights[i-1]+attention_weights[i]

    return attention

def highest_x(a,w,p=1):
    """
    Inputs:
        a: a 1-D numpy array contains the scores of each position
        w: length of window to aggregate the scores
        p: length of padding when maximum sum of consecutive numbers are taken
    """

    lists = [{k:v for (k,v) in zip(range(len(a)),a)}]
    result = {}
    max_idx = len(a) -1
    count = 1
    condition = [True]
    while any(con is True for con in condition):
        starts = []
        ends = []
        bests = []

        for ele in lists:
            values = list(ele.values())
            idx = list(ele.keys())


            start_idx = idx[0]

            if len(values) >= w:
                highest, highest_idx_start, highest_idx_end = highest_score(values,w)

                starts.append(highest_idx_start+start_idx)


                ends.append(highest_idx_end+start_idx)


                bests.append(highest)


        best_idx = max(zip(bests, range(len(bests))))[1]   # calculate the index of maximum sum

        cut_value = bests[best_idx]

        if starts[best_idx] - p >=0:
            cut_idx_start = starts[best_idx] - p
        else:
            cut_idx_start = 0

        if ends[best_idx] + p <=max_idx:
            cut_idx_end = ends[best_idx] + p
        else:
            cut_idx_end = max_idx

        result[count] = (cut_value,starts[best_idx],ends[best_idx])


        copy = lists.copy()

        for ele in lists:
            values = list(ele.values())
            idx = list(ele.keys())

            start_idx, end_idx = idx[0], idx[-1]

            if len(values) < w:
                copy.remove(ele)
            else:
#                 print(cut_idx_start,cut_idx_end)
#                 print(start_idx,end_idx)
#                 print(values)
                if (cut_idx_end < start_idx) or (cut_idx_start > end_idx):

                    pass
                elif (cut_idx_start < start_idx) and (cut_idx_end >= start_idx):
                    copy.remove(ele)
                    values = values[cut_idx_end-start_idx+1:]
                    idx = idx[cut_idx_end-start_idx+1:]
                    ele = {k:v for (k,v) in zip(idx,values)}

                    if ele != {}:
                        copy.append(ele)

                elif (cut_idx_start >= start_idx) and (cut_idx_end <= end_idx):
                    copy.remove(ele)
                    values_1 = values[:cut_idx_start-start_idx]
                    idx_1 = idx[:cut_idx_start-start_idx]
                    ele_1 = {k:v for (k,v) in zip(idx_1,values_1)}

                    values_2 = values[cut_idx_end-start_idx+1:]
                    idx_2 = idx[cut_idx_end-start_idx+1:]
                    ele_2 = {k:v for (k,v) in zip(idx_2,values_2)}

                    if ele_1 != {}:
                        copy.append(ele_1)
                    if ele_2 != {}:
                        copy.append(ele_2)

                elif (cut_idx_start <= end_idx) and (cut_idx_end > end_idx):
                    copy.remove(ele)
                    values = values[:cut_idx_start-start_idx]
                    idx = idx[:cut_idx_start-start_idx]
                    ele = {k:v for (k,v) in zip(idx,values)}

                    if ele != {}:
                        copy.append(ele)

        lists = copy
#        print(lists)
        count = count + 1
        condition = [len(i)>=w for i in lists]
#        print(condition)

    return result

def highest_score(a,w):
    """
    Inputs:
        a: a 1-D numpy array contains the scores of each position
        w: length of window to aggregate the scores
    """

    assert(len(a)>=w)

    best = -20000
    best_idx_start = 0
    best_idx_end =0
    for i in range(len(a)-w + 1):
        tmp = np.sum(a[i:i+w])
        if tmp > best:
            best = tmp
            best_idx_start = i
            best_idx_end = i + w - 1

    return best, best_idx_start, best_idx_end