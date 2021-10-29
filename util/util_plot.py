import matplotlib.pyplot as plt
import seaborn as sns
import collections
import umap
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

colors = ["#8D5CDC", "#EA52BC", "#FF6691", "#FF946B","#FFC859", "#F9F871"]

def draw_protein_residues_hist_image(train_data,trainlabel,test_data,config):
    # train_data 和 test_data 要保证有sequences 和 labels成员属性
    keyvalueP = {}
    keyvalueF = {}
    keyvalueP['A'] = 0
    keyvalueP['R'] = 0
    keyvalueP['N'] = 0
    keyvalueP['D'] = 0
    keyvalueP['C'] = 0
    keyvalueP['Q'] = 0
    keyvalueP['E'] = 0
    keyvalueP['G'] = 0
    keyvalueP['H'] = 0
    keyvalueP['I'] = 0
    keyvalueP['L'] = 0
    keyvalueP['K'] = 0
    keyvalueP['M'] = 0
    keyvalueP['F'] = 0
    keyvalueP['P'] = 0
    keyvalueP['S'] = 0
    keyvalueP['T'] = 0
    keyvalueP['W'] = 0
    keyvalueP['Y'] = 0
    keyvalueP['V'] = 0

    keyvalueF['A'] = 0
    keyvalueF['R'] = 0
    keyvalueF['N'] = 0
    keyvalueF['D'] = 0
    keyvalueF['C'] = 0
    keyvalueF['Q'] = 0
    keyvalueF['E'] = 0
    keyvalueF['G'] = 0
    keyvalueF['H'] = 0
    keyvalueF['I'] = 0
    keyvalueF['L'] = 0
    keyvalueF['K'] = 0
    keyvalueF['M'] = 0
    keyvalueF['F'] = 0
    keyvalueF['P'] = 0
    keyvalueF['S'] = 0
    keyvalueF['T'] = 0
    keyvalueF['W'] = 0
    keyvalueF['Y'] = 0
    keyvalueF['V'] = 0

    sequeces = train_data.sequences
    labels = train_data.labels
    for i in range(len(sequeces)):
        seq = sequeces[i]
        label = labels[i]
        if (label == 1):  # 正例
            for word in seq:
                if word not in keyvalueP.keys():
                    keyvalueP[word] = 1
                else:
                    keyvalueP[word] += 1
        else:
            for word in seq:
                if word not in keyvalueF.keys():
                    keyvalueF[word] = 1
                else:
                    keyvalueF[word] += 1
    fig, ax = plt.subplots()
    title = "The number of different protein residues(Train)"
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.title(title)
    plt.xlabel("The types of protein residues", fontsize='x-large', fontweight='light')
    plt.ylabel("The number of protein residues", fontsize='x-large', fontweight='light')
    # 拿到数据
    Px, Py, Nx, Ny = list(keyvalueP.keys()), list(keyvalueP.values()), \
           list(keyvalueF.keys()), list(keyvalueF.values())
    # 根据坐标画柱状图
    ind = np.arange(20) + 20
    width = 0.4
    plt.bar(ind, Py, color='#b5cce4', edgecolor='#929195', width=width, alpha=1.0, label="Positive")
    plt.bar(ind + width, Ny, color='#f9f8fd', edgecolor='#929195', width=width, alpha=0.50, label="Negative")
    plt.legend(loc="upper left", )
    plt.legend(frameon=False)
    plt.xticks(np.arange(20) + 20 + 0.2, ('A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G',
                                          'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S',
                                          'T', 'W', 'Y', 'V'),
               fontsize=11)
    plt.savefig('{}/{}/{}.{}'.format(config.path_save, config.learn_name, 'statistics', config.save_figure_type))
    plt.show()

    keyvalueP['A'] = 0
    keyvalueP['R'] = 0
    keyvalueP['N'] = 0
    keyvalueP['D'] = 0
    keyvalueP['C'] = 0
    keyvalueP['Q'] = 0
    keyvalueP['E'] = 0
    keyvalueP['G'] = 0
    keyvalueP['H'] = 0
    keyvalueP['I'] = 0
    keyvalueP['L'] = 0
    keyvalueP['K'] = 0
    keyvalueP['M'] = 0
    keyvalueP['F'] = 0
    keyvalueP['P'] = 0
    keyvalueP['S'] = 0
    keyvalueP['T'] = 0
    keyvalueP['W'] = 0
    keyvalueP['Y'] = 0
    keyvalueP['V'] = 0

    keyvalueF['A'] = 0
    keyvalueF['R'] = 0
    keyvalueF['N'] = 0
    keyvalueF['D'] = 0
    keyvalueF['C'] = 0
    keyvalueF['Q'] = 0
    keyvalueF['E'] = 0
    keyvalueF['G'] = 0
    keyvalueF['H'] = 0
    keyvalueF['I'] = 0
    keyvalueF['L'] = 0
    keyvalueF['K'] = 0
    keyvalueF['M'] = 0
    keyvalueF['F'] = 0
    keyvalueF['P'] = 0
    keyvalueF['S'] = 0
    keyvalueF['T'] = 0
    keyvalueF['W'] = 0
    keyvalueF['Y'] = 0
    keyvalueF['V'] = 0

    sequeces = test_data.sequences
    labels = test_data.labels
    for i in range(len(sequeces)):
        seq = sequeces[i]
        label = labels[i]
        if (label == 1):  # 正例
            for word in seq:
                if word not in keyvalueP.keys():
                    keyvalueP[word] = 1
                else:
                    keyvalueP[word] += 1
        else:
            for word in seq:
                if word not in keyvalueF.keys():
                    keyvalueF[word] = 1
                else:
                    keyvalueF[word] += 1
    fig, ax = plt.subplots()
    title = "The number of different protein residues(Test)"
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.title(title)
    plt.xlabel("The types of protein residues", fontsize='x-large', fontweight='light')
    plt.ylabel("The number of protein residues", fontsize='x-large', fontweight='light')
    # 拿到数据
    Px, Py, Nx, Ny = list(keyvalueP.keys()), list(keyvalueP.values()), \
                     list(keyvalueF.keys()), list(keyvalueF.values())
    # 根据坐标画柱状图
    ind = np.arange(20) + 20
    width = 0.4
    plt.bar(ind, Py, color='#b5cce4', edgecolor='#929195', width=width, alpha=1.0, label="Positive")
    plt.bar(ind + width, Ny, color='#f9f8fd', edgecolor='#929195', width=width, alpha=0.50, label="Negative")
    plt.legend(loc="upper left", )
    plt.legend(frameon=False)
    plt.xticks(np.arange(20) + 20 + 0.2, ('A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G',
                                          'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S',
                                          'T', 'W', 'Y', 'V'),
               fontsize=11)
    plt.savefig('{}/{}/{}.{}'.format(config.path_save, config.learn_name, 'statistics', config.save_figure_type))
    plt.show()

def draw_base_hist_image(train_data,test_data,config):

    keyvalueP = {}
    keyvalueF = {}
    keyvalueP['A'] = 0
    keyvalueP['T'] = 0
    keyvalueP['C'] = 0
    keyvalueP['G'] = 0

    keyvalueF['A'] = 0
    keyvalueF['T'] = 0
    keyvalueF['C'] = 0
    keyvalueF['G'] = 0

    sequeces = train_data.sequences
    labels = train_data.labels

    for i in range(len(sequeces)):
        seq = sequeces[i]
        label = labels[i]

        if(label == 1): # 正例
            for word in seq:
                if word not in keyvalueP.keys():
                    keyvalueP[word] = 1
                else :
                    keyvalueP[word]+=1
        else:
            for word in seq:
                if word not in keyvalueF.keys():
                    keyvalueF[word] = 1
                else :
                    keyvalueF[word]+=1
    fig, ax = plt.subplots()
    title = "The number of different base(Train)"
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.title(title)
    plt.xlabel("The types of base", fontsize='x-large', fontweight='light')
    plt.ylabel("The number of base", fontsize='x-large', fontweight='light')
    # 拿到数据
    Px, Py, Nx, Ny = list(keyvalueP.keys()), list(keyvalueP.values()), \
                     list(keyvalueF.keys()), list(keyvalueF.values())
    # 根据坐标画柱状图
    ind = np.arange(4) + 20
    width = 0.4
    plt.bar(ind, Py, color='#b5cce4', edgecolor='#929195', width=width, alpha=1.0, label="Positive")
    plt.bar(ind + width, Ny, color='#f9f8fd', edgecolor='#929195', width=width, alpha=0.50, label="Negative")
    plt.legend(loc="upper left", )
    plt.legend(frameon=False)
    plt.xticks(np.arange(4) + 20 + 0.2, ('A', 'T', 'C', 'G'),
               fontsize=11)
    plt.savefig('{}/{}/{}.{}'.format(config.path_save, config.learn_name, 'statistics', config.save_figure_type))
    plt.show()

    keyvalueP = {}
    keyvalueF = {}
    keyvalueP['A'] = 0
    keyvalueP['T'] = 0
    keyvalueP['C'] = 0
    keyvalueP['G'] = 0

    keyvalueF['A'] = 0
    keyvalueF['T'] = 0
    keyvalueF['C'] = 0
    keyvalueF['G'] = 0

    sequeces = test_data.sequences
    labels = test_data.labels

    for i in range(len(sequeces)):
        seq = sequeces[i]
        label = labels[i]

        if (label == 1):  # 正例
            for word in seq:
                if word not in keyvalueP.keys():
                    keyvalueP[word] = 1
                else:
                    keyvalueP[word] += 1
        else:
            for word in seq:
                if word not in keyvalueF.keys():
                    keyvalueF[word] = 1
                else:
                    keyvalueF[word] += 1
    fig, ax = plt.subplots()
    title = "The number of different base(Test)"
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.title(title)
    plt.xlabel("The types of base", fontsize='x-large', fontweight='light')
    plt.ylabel("The number of base", fontsize='x-large', fontweight='light')
    # 拿到数据
    Px, Py, Nx, Ny = list(keyvalueP.keys()), list(keyvalueP.values()), \
                     list(keyvalueF.keys()), list(keyvalueF.values())
    # 根据坐标画柱状图
    ind = np.arange(4) + 20
    width = 0.4
    plt.bar(ind, Py, color='#b5cce4', edgecolor='#929195', width=width, alpha=1.0, label="Positive")
    plt.bar(ind + width, Ny, color='#f9f8fd', edgecolor='#929195', width=width, alpha=0.50, label="Negative")
    plt.legend(loc="upper left", )
    plt.legend(frameon=False)
    plt.xticks(np.arange(4) + 20 + 0.2, ('A', 'T', 'C', 'G'),
               fontsize=11)
    plt.savefig('{}/{}/{}.{}'.format(config.path_save, config.learn_name, 'statistics', config.save_figure_type))
    plt.show()

def draw_dna_rna_length_distribution_image(train_data,test_data,config):
    # train_data 中是处理之后的训练集中的正负例样本长度集合
    # test_data 中是处理之后的测试集中的正负例样本长度集合
    # 或者在这个方法里用循环处理数据
    data1 = train_data.positive_lengths
    data2 = train_data.negative_lengths
    xlabel = 'Length'
    ylabel = 'Number'
    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.legend(loc="upper left", )
    plt.title('The length of sequences(Train)')
    plt.xlabel(xlabel, fontsize='x-large', fontweight='light')
    plt.ylabel(ylabel, fontsize='x-large', fontweight='light')
    plt.hist(data1, bins=10, edgecolor='black', alpha=1, color='#b5cce4', histtype='bar',
             align='mid', orientation='vertical', rwidth=None, log=False, label='positive', stacked=False, )
    plt.hist(data2, bins=10, edgecolor='black', alpha=0.50, color='#f9f8fd', histtype='bar',
             align='mid', orientation='vertical', rwidth=None, log=False, label='negative', stacked=False, )
    plt.legend(frameon=False)
    plt.savefig('{}/{}/{}.{}'.format(config.path_save, config.learn_name, 'statistics', config.save_figure_type))
    plt.show()

    data1 = test_data.positive_lengths
    data2 = test_data.negative_lengths
    xlabel = 'Length'
    ylabel = 'Number'
    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.legend(loc="upper left", )
    plt.title('The length of sequences(Test)')
    plt.xlabel(xlabel, fontsize='x-large', fontweight='light')
    plt.ylabel(ylabel, fontsize='x-large', fontweight='light')
    plt.hist(data1, bins=10, edgecolor='black', alpha=1, color='#b5cce4', histtype='bar',
             align='mid', orientation='vertical', rwidth=None, log=False, label='positive', stacked=False, )
    plt.hist(data2, bins=10, edgecolor='black', alpha=0.50, color='#f9f8fd', histtype='bar',
             align='mid', orientation='vertical', rwidth=None, log=False, label='negative', stacked=False, )
    plt.legend(frameon=False)
    plt.savefig('{}/{}/{}.{}'.format(config.path_save, config.learn_name, 'statistics', config.save_figure_type))
    plt.show()

def draw_ROC_PRC_curve(roc_datas, prc_datas, name, config):
    # roc_data = [FPR, TPR, AUC]
    # prc_data = [recall, precision, AP]

    sns.set(style="darkgrid")
    plt.figure(figsize=(16, 8))
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    lw = 2

    plt.subplot(1, 2, 1)
    for index, roc_data in enumerate(roc_datas):
            plt.plot(roc_data[0], roc_data[1], color=colors[index],lw=lw, label=name[index] + ' (AUC = %0.2f)' % roc_data[2])  ### 假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontdict={'weight': 'normal', 'size': 20})
    plt.ylabel('True Positive Rate', fontdict={'weight': 'normal', 'size': 20})
    plt.title('receiver operating characteristic curve', fontdict={'weight': 'normal', 'size': 23})
    plt.legend(loc="lower right", prop={'weight': 'normal', 'size': 16})

    plt.subplot(1, 2, 2)
    # plt.step(self.prc_data[0], self.prc_data[1], color='b', alpha=0.2,where='post')
    # plt.fill_between(prc_data[0], prc_data[1], step='post', alpha=0.2,color='b')
    for index, prc_data in enumerate(prc_datas):
        plt.plot(prc_data[0], prc_data[1], color=colors[index],
             lw=lw, label=name[index] + ' (AP = %0.2f)' % prc_data[2])  ### 假正率为横坐标，真正率为纵坐标做曲线

    plt.xlabel('Recall', fontdict={'weight': 'normal', 'size': 20})
    plt.ylabel('Precision', fontdict={'weight': 'normal', 'size': 20})
    plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision Recall curve', fontdict={'weight': 'normal', 'size': 23})
    plt.legend(loc="lower left", prop={'weight': 'normal', 'size': 16})

    plt.savefig(
        '{}/{}/{}.{}'.format(config.path_save , config.learn_name , 'ROC_PRC', config.save_figure_type))
    # plt.show()

def draw_statistics_bar(traindataset, testdataset, config):
    totaldataset = []
    totaldataset.extend(traindataset)
    totaldataset.extend(testdataset)

    plt.figure()

    # Todo
    if config.model == 'DNAbert':
        colors = ['#4DE199', '#F4E482', '#BAAD4E','#827919']
        statstic = [0,0,0,0] # A, C, T, G
        labels = ['A', 'C', 'T', 'G']
        for seq in totaldataset:
            for i in range(len(seq)):
                if seq[i] == 'A':
                    statstic[0] = statstic[0] + 1
                elif seq[i] == 'C':
                    statstic[1] = statstic[1] + 1
                elif seq[i] == 'T':
                    statstic[2] = statstic[2] + 1
                elif seq[i] == 'G':
                    statstic[3] = statstic[3] + 1
        # print(statstic)
        plt.bar(labels, statstic, color=colors)  # or `color=['r', 'g', 'b']`
    elif config.model == 'prot_bert_bfd' or config.model == 'prot_bert':
        colors = ['#e52d5d', '#e01b77', '#d31b92', '#bb2cad', '#983fc5', '#7b60e0', '#547bf3', '#0091ff', '#00b6ff', '#00d0da', '#00df83', '#a4e312' ]
        statstic = collections.defaultdict(int)
        for seq in totaldataset:
            for i in seq:
                statstic[i] += 1
        # print(statstic)
        labels = statstic.keys()
        plt.bar(statstic.keys(), statstic.values(), color=colors[:len(labels)])  # or `color=['r', 'g', 'b']`

    plt.savefig('{}/{}/{}.{}'.format(config.path_save, config.learn_name, 'statistics', config.save_figure_type))

    plt.show()

def draw_umap(repres_list,label_list):
    #配色、名称、坐标轴待统一样式修改
    repres=np.array(repres_list)
    label = np.array(label_list)
    scaled_data = StandardScaler().fit_transform(repres)
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(scaled_data)
    # print(embedding)
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=label, cmap='summer', s=5
    )
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection ', fontsize=24)
    plt.show()

def draw_negative_density(logist_list, label_list):
    plt.figure(figsize=(30, 15))
    fig, ax = plt.subplots()
    # sns.kdeplot(len_ls[y_ls == 0], shade=True, alpha=.25, label='model A', common_norm=False, color='#CC3366')
    # sns.kdeplot(len_ls1[y_ls == 0], shade=True, alpha=.25, label='model B', color='#00CCCC')
    sns.kdeplot(logist_list[label_list == 0], shade=True, alpha=.25, label='model C', color='#66CC00')
    ax.vlines(0.25, 0, 0.8, colors='#999999', linestyles="dashed")
    ax.vlines(0.75, 0, 1.2, colors='#999999', linestyles="dashed")
    ax.tick_params(direction='out', labelsize=12)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(-0.25, 1.25)
    ax.set_xticks([-0.25, 0.05, 0.35, 0.65, 0.95, 1.25])
    ax.set_xticklabels(('0.0', '0.2', '0.4', '0.6', '0.8', '1.0'))
    ax.set_ylim(0, 1.4)
    plt.ylabel('Density (predicted negative samples)', fontsize=12)
    plt.xlabel('Confidence', fontsize=12)

    plt.legend(bbox_to_anchor=(0.1, 1.4), loc=2, borderaxespad=0, numpoints=1, fontsize=12, frameon=False)
    fig.subplots_adjust(top=0.7, left=0.2)
    plt.tight_layout()
    plt.show()

def draw_positive_density(logist_list, label_list):
    plt.figure(figsize=(30, 15))
    fig, ax = plt.subplots()
    # sns.kdeplot(len_ls[y_ls == 0], shade=True, alpha=.25, label='model A', common_norm=False, color='#CC3366')
    # sns.kdeplot(len_ls1[y_ls == 0], shade=True, alpha=.25, label='model B', color='#00CCCC')
    sns.kdeplot(logist_list[label_list == 1], shade=True, alpha=.25, label='model C', color='#66CC00')
    ax.vlines(0.25, 0, 0.8, colors='#999999', linestyles="dashed")
    ax.vlines(0.75, 0, 1.2, colors='#999999', linestyles="dashed")
    ax.tick_params(direction='out', labelsize=12)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(-0.25, 1.25)
    ax.set_xticks([-0.25, 0.05, 0.35, 0.65, 0.95, 1.25])
    ax.set_xticklabels(('0.0', '0.2', '0.4', '0.6', '0.8', '1.0'))
    ax.set_ylim(0, 1.4)
    plt.ylabel('Density (predicted positive samples)', fontsize=12)
    plt.xlabel('Confidence', fontsize=12)

    plt.legend(bbox_to_anchor=(0.1, 1.4), loc=2, borderaxespad=0, numpoints=1, fontsize=12, frameon=False)
    fig.subplots_adjust(top=0.7, left=0.2)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    c = ['DLIPTSSKLVV','DLIPTSSKLVV','DLIPTSSKLVV','AETCZAO','ABCDEFGHIJKMN']
    colors = ['#e52d5d', '#e01b77', '#d31b92', '#bb2cad', '#983fc5', '#7b60e0', '#547bf3', '#0091ff', '#00b6ff', '#00d0da', '#00df83', '#a4e312' ]
    statstic = collections.defaultdict(int)
    for seq in c:
        for i in seq:
            statstic[i] += 1
    # print(statstic)
    labels = statstic.keys()
    s = statstic.values()
    print(plt.cm.get_cmap('rainbow', len(labels)))
    plt.bar(labels, s, color=colors[:len(labels)])  # or `color=['r', 'g', 'b']`
    plt.show()
