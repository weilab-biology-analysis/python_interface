import numpy as np
from traditional_desc import nu_extract
from traditional_desc import protein_extract

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from util import util_file

def main(featurechoice, config):
    nucleotide_feature = ['ANF', 'binary', 'CKSNAP', 'DNC']
    protein_feature = ['BLOSUM62', 'CKSAAGP', 'CTDC']

    ROCdatas = []
    PRCdatas = []

    # Import some data to play with
    datapath = config.path_data

    # train_data用于训练的样本集, test_data用于测试的样本集, train_label训练样本对应的标签集, test_label测试样本对应的标签集
    train_seq, train_label, test_seq, test_label = util_file.load_fasta(datapath)
    train_data, test_data = [], []

    if config.type == 'DNA' and 'RNA':
        for i in featurechoice:
            feature = nucleotide_feature[i]
            if feature == 'ANF':
                train_data = nu_extract.ANF(train_seq)
                test_data = nu_extract.ANF(test_seq)
            if feature == 'binary':
                train_data = nu_extract.binary(train_seq)
                test_data = nu_extract.binary(test_seq)
            if feature == 'CKSNAP':
                train_data = nu_extract.CKSNAP(train_seq)
                test_data = nu_extract.CKSNAP(test_seq)
            if feature == 'DNC':
                train_data = nu_extract.DNC(train_seq)
                test_data = nu_extract.DNC(test_seq)
            ROCdata, PRCdata = gen(train_data, train_label, test_data, test_label)
            ROCdatas.append(ROCdata)
            PRCdatas.append(PRCdata)

    elif config.type == 'protein':
        for i in featurechoice:
            feature = protein_feature[i]
            if feature == 'BLOSUM62':
                train_data = protein_extract.BLOSUM62(train_seq)
                test_data = protein_extract.BLOSUM62(test_seq)
            elif feature == 'CKSNAP':
                train_data = protein_extract.CKSAAGP(train_seq)
                test_data = protein_extract.CKSAAGP(test_seq)
            elif feature == 'CTDC':
                train_data = protein_extract.CTDC(train_seq)
                test_data = protein_extract.CTDC(test_seq)
            ROCdata, PRCdata = gen(train_data, train_label, test_data, test_label)
            ROCdatas.append(ROCdata)
            PRCdatas.append(PRCdata)

    return ROCdatas, PRCdatas

def gen(train_data, train_label, test_data, test_label):

    # Learn to predict each class against the other分类器设置
    from sklearn import svm
    random_state = np.random.RandomState(0)
    svm = svm.SVC(kernel='linear', probability=True, random_state=random_state)  # 使用核函数为线性核，参数默认，创建分类器

    ###通过decision_function()计算得到的test_predict_label的值，用在roc_curve()函数中
    test_predict_label = svm.fit(train_data, train_label).decision_function(test_data)
    # 首先通过fit来对训练样本和训练样本标签进行训练得到模型，然后通过decision_function来获得模型对于测试样本集预测的标签集
    # print(test_predict_label)

    # Compute ROC curve and ROC area for each class#计算tp,fp
    # 通过测试样本输入的标签集和模型预测的标签集进行比对，得到fp,tp,不同的fp,tp是算法通过一定的规则改变阈值获得的

    # ROC画图
    fpr, tpr, threshold = roc_curve(test_label, test_predict_label)  ###计算真正率和假正率
    roc_auc = auc(fpr,tpr)
    print(roc_auc)
    # PRC画图
    precision, recall, thresholds = precision_recall_curve(test_label, test_predict_label, pos_label=1)
    AP = average_precision_score(test_label, test_predict_label, average='macro', pos_label=1, sample_weight=None)

    return [fpr, tpr, roc_auc], [precision, recall, AP]