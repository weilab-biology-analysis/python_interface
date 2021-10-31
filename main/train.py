import sys
import os
import pickle
import json
import zipfile
import torch

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import requests
from configuration import config_init
from frame import Learner
from util import util_plot, util_json, util_file
from traditional_desc import generate


def SL_train(config, modelsORpara):
    torch.cuda.set_device(config.device)
    roc_datas, prc_datas = [], []
    repres_list, label_list = [], []
    train_seq, test_seq = [], []
    train_label, test_label = [], []
    pos_list = []
    neg_list = []
    if_same = config.if_same
    if_same = True
    config.if_same = if_same
    savepath = '../data/result/' + config.learn_name
    if not os.path.exists('../data/result/'):
        os.mkdir('../data/result/')
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    util_file.filiter_fasta(config.path_data, savepath, skip_first=False)

    names = []
    if config.minimode == 'paramCompare':
        for i in modelsORpara:
            names.append(config.model + '_' + config.para + '_' + i)

    elif config.minimode == 'modelCompare':
        names = modelsORpara
        for index, name in enumerate(names):
            if name in ["3mer_DNAbert", "4mer_DNAbert", "5mer_DNAbert", "6mer_DNAbert"]:
                names[index] = 'DNAbert'

    tra_ROCdatas, tra_PRCdatas, tra_name = generate.main([0, 1, 2], config)

    plot_config = {'names': names,
                   'tra_name': tra_name,
                   'model_number': len(modelsORpara),
                   'savepath': savepath + '/plot',
                   'type': config.type,
                   'if_same': config.if_same,
                   'fasta_list': [savepath + '/train_positive.txt', savepath + '/train_negative.txt',
                                  savepath + '/test_positive.txt', savepath + '/test_negative.txt']}
    # tra_ROCdatas, tra_PRCdatas = generate.main([0, 1, 2], config)

    for index, model in enumerate(modelsORpara):
        # print(models)
        if config.minimode == 'paramCompare':
            if config.para == "CDHit":
                outputpath = '/data/result' + config.learn_name + '/' + "CDhit_" + str(
                    modelsORpara[index]) + "_data.fasta"
                cmd = "/home/wrh/weilab_server/cdhit-4.6.2/cd-hit -i " + config.origin_path_data + " -o " + outputpath + " -c " + str(
                    modelsORpara[index]) + " -aS 0.8 -d 0"
                os.system(cmd)
                config.path_data = outputpath

        elif config.minimode == 'modelCompare':
            config.model = model

        learner = Learner.Learner(config)
        learner.setIO()
        learner.setVisualization()
        learner.SL_train_load_data()
        learner.init_model()
        # learner.load_params()
        learner.init_optimizer()
        learner.def_loss_func()
        learner.train_model()

        if index == 0:
            print("save traindata and testdata")
            train_seq = learner.dataManager.train_dataset
            test_seq = learner.dataManager.test_dataset
            train_label = learner.dataManager.train_label
            test_label = learner.dataManager.test_label
            train_data = [train_seq, train_label]
            test_data = [test_seq, test_label]
            # util_plot.draw_statistics_bar(learner.dataManager.train_dataset, learner.dataManager.test_dataset, config)

        # plot data prepare
        roc_datas.append(learner.visualizer.roc_data)
        prc_datas.append(learner.visualizer.prc_data)
        repres_list.append(learner.visualizer.repres_list)
        pos_list.append(learner.visualizer.pos_list)
        neg_list.append(learner.visualizer.neg_list)
        label_list.append(learner.visualizer.label_list)
        # plot_config['name'].append(str(model))

    # print("logits_list1: ", logits_list)
    plot_data = {'train_data': train_data,
                 'test_data': test_data,
                 'repres_list': repres_list,
                 'pos_list': pos_list,
                 'neg_list': neg_list,
                 'label_list': label_list,
                 'roc_datas': roc_datas,
                 'prc_datas': prc_datas,
                 'config': plot_config
                 }

    util_plot.plot_interface(plot_data)

    # drow(plot_data)
    # print("plot_data_save")
    # torch.save(plot_data, './plot_data.pth')
    # print("plot_data_complete")

    # util_plot.draw_ROC_PRC_curve(roc_datas, prc_datas, name, config)
    # learner.test_model()

    # return data

def SL_test():
    # config = config_SL.get_config()
    config = pickle.load(open('../result/jobID/config.pkl', 'rb'))
    config.path_params = '../result/jobID/DNAbert, MCC[0.64].pt'
    learner = Learner.Learner(config)
    learner.setIO()
    learner.setVisualization()
    learner.load_data()
    learner.init_model()
    learner.load_params()
    learner.init_optimizer()
    learner.def_loss_func()
    learner.train_model()
    learner.test_model()


def gpu_test():
    config = config_init.get_config()

    # input_fa_file = config.path_data
    # str_list = input_fa_file.split("/")
    # out_fa_file = "/"
    # for i in range(1, len(str_list) - 1):
    #     # out_fa_file =
    #     out_fa_file = out_fa_file + str_list[i] + "/"
    # cmd = "/home/wrh/weilab_server/cdhit-4.6.2/cd-hit -i " + input_fa_file + " -o " + out_fa_file + "new.fasta -c 0.8 -aS 0.8 -d 0"
    #
    # os.system(cmd)
    # print("aaaaaaaaaaa:", a)
    # config.path_data = out_fa_file + "new.fasta"
    # motif = "/home/weilab/anaconda3/envs/wy/bin/weblogo --format PNG -f " + config.path_data + " -o test4.png"
    # os.system(motif)
    config.mode = "train-test"
    config.minimode = "modelCompare"
    config.type = "DNA"
    config.if_same = True
    # SL_train(config, ["LSTM", "TransformerEncoder"])
    SL_train(config, ["TransformerEncoder"])
    # SL_train(config, ["TextRCNN"])
    # SL_train(config, ["TextGCN"])
    # SL_train(config, ["TextRCNN", "BiLSTM", "6mer_DNAbert", "LSTM", "VDCNN", "LSTMAttention", "Reformer_Encoder", "Performer_Encoder"])


def server_use():
    model_all = ["DNN", "RNN", "LSTM", "BiLSTM", "LSTMAttention", "GRU", "TextCNN", "TextRCNN", "VDCNN", "RNN_CNN",
                 "TransformerEncoder", "Reformer_Encoder", "Performer_Encoder", "Linformer_Encoder",
                 "RoutingTransformer_Encoder", "6mer_DNAbert", "prot_bert", "TextGCN"]
    para_all = ["CDHit", "Focalloss", "adversialtraining"]
    CDHit_all = [0.8, 0.6, 0.4]
    # Focalloss_all = [0, 1]
    # adversialtraining_all = [0,1]

    # RNA_model = ["TransformerEncoder", "TextCNN", "LSTM", "GCN"]
    # prot_model = ["prot_bert", "TransformerEncoder", "TextCNN", "LSTM", "GCN"]

    os.chdir("/home/weilab/molecular_analysis_server/biology_python/main")
    # print(sys.argv[2])
    # print(type(sys.argv[2]))
    setting = json.loads(sys.argv[2])
    requests_url = "http://server.wei-group.net/biology/job/status/update"

    config = config_init.get_config()
    config.learn_name = str(setting["jobId"])
    config.path_data = config.origin_path_data = setting["requestDataPath"]
    config.path_save = setting["resultDataPath"]

    config.mode = setting["mode"]
    config.minimode = setting["minimode"]
    config.type = setting["type"]
    config.if_same = True

    os.mkdir(config.path_save + '/' + config.learn_name)

    # a = '01234'
    # list(a)
    # Out[6]: ['0', '1', '2', '3', '4']
    if config.minimode == 'paramCompare':
        config.paramCompare = setting['paramCompare']
        config.model = model_all[int(setting['paramCompareModel'])]
        if config.paramCompare == "CDHit":
            CDHitlist = list(setting['CDHit'])
            models = [i / 10 for i in CDHitlist]

    elif config.minimode == 'modelCompare':
        modelchoice = setting["modelCompare"].split(' ')
        models = []

        for choice in modelchoice:
            models.append(model_all[int(choice)])

    requests.post(requests_url, util_json.get_json(config.learn_name, 1))

    try:
        SL_train(config, models)

        # resultdir = '/data/result/' + config.learn_name
        # zip_file = zipfile.ZipFile(resultdir + 'zipdir' + '.zip', 'w')
        # # 把zfile整个目录下所有内容，压缩为new.zip文件
        # zip_file.write(resultdir, compress_type=zipfile.ZIP_DEFLATED)
        # zip_file.close()

        # pictureArray = {}

        path_data = "http://server.wei-group.net" + '/result/' + config.learn_name + '/plot/'

        for root, dirs, files in os.walk(config.path_save + '/' + config.learn_name + '/plot'):
            print("files", files)  # 当前路径下所有非目录子文件
            pictureArray = [path_data + i for i in files ]

        #
        # for model in models:
        #     pictureArray.append(
        #         "http://server.wei-group.net" + '/result/' + config.learn_name + '/' + model + 'mer/t-sne.png')

        result = {
            # "zip": "http://server.wei-group.net" + '/result/' + config.learn_name + 'zipdir' + '.zip',
            "pictures": pictureArray
        }

        postget = requests.post(requests_url, util_json.get_json(config.learn_name, 2, json.dumps(result)))
        print(postget.text)
    except Exception as e:
        print(e)
        requests.post(requests_url, util_json.get_json(config.learn_name, -1))

    # SL_train(config, models)


if __name__ == '__main__':
    # server_use()
    gpu_test()
