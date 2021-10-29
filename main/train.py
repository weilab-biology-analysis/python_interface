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
from util import util_plot, util_json
from traditional_desc import generate

def SL_train(config, models):
    torch.cuda.set_device(config.device)
    roc_datas, prc_datas = [], []
    repres_list, label_list = [], []
    train_data,test_data = [], []
    plot_config = { 'name':[],
                    'length': len(models),
                    'savepath': '/data/result/' + config.learn_name}

    for index, model in enumerate(models):
        config.model = model
        learner = Learner.Learner(config)
        learner.setIO()
        learner.setVisualization()
        learner.load_data()
        learner.init_model()
        # learner.load_params()
        learner.init_optimizer()
        learner.def_loss_func()
        learner.train_model()

        if index == 0:
            print("save traindata and testdata")
            train_data = learner.dataManager.train_dataset
            test_data = learner.dataManager.test_dataset
            # util_plot.draw_statistics_bar(learner.dataManager.train_dataset, learner.dataManager.test_dataset, config)

        # plot data prepare
        roc_datas.append(learner.visualizer.roc_data)
        prc_datas.append(learner.visualizer.prc_data)
        repres_list.append(learner.visualizer.repres_list)
        label_list.append(learner.visualizer.label_list)
        plot_config['name'].append(str(model))


    plot_data = {'train_data':train_data,
             'test_data':test_data,
             'repres_list':repres_list,
             'label_list': label_list,
             'roc_datas': roc_datas,
             'prc_datas': prc_datas,
             'config': plot_config

             }
    # print("plot_data_save")
    # torch.save(state, './plot_data.pth')
    # print("plot_data_complete")

    # util_plot.draw_ROC_PRC_curve(roc_datas, prc_datas, name, config)
    # learner.test_model()

    # return data

def traditional_train(config):
    generate.main([3], config)


def SL_fintune():
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
    # SL_train(config, ["BiLSTM"])
    # SL_train(config, ["RNN"])
    # SL_train(config, ["TextRCNN", "BiLSTM", "6mer_DNAbert", "LSTM", "VDCNN", "LSTMAttention", "Reformer_Encoder", "Performer_Encoder"])
    # SL_train(config, ["TextRCNN"])
    # SL_train(config, ["TextGCN"])
    traditional_train(config)

def server_use():
    DNA_model = ["3mer_DNAbert", "4mer_DNAbert", "5mer_DNAbert", "6mer_DNAbert", "TransformerEncoder", "TextCNN", "LSTM", "GCN"]
    RNA_model = ["TransformerEncoder", "TextCNN", "LSTM", "GCN"]
    prot_model = ["prot_bert", "TransformerEncoder", "TextCNN", "LSTM", "GCN"]
    os.chdir("/root/biology_python/main")
    # print(sys.argv[2])
    # print(type(sys.argv[2]))
    setting = json.loads(sys.argv[2])
    requests_url = "http://server.wei-group.net/biology/job/status/update"

    config = config_init.get_config()
    config.learn_name = str(setting["jobId"])
    config.path_data = setting["requestDataPath"]
    config.path_save = setting["resultDataPath"]
    config.type = setting["type"]

    modelchoice = setting["modelchoice"]
    models = []

    # Todo
    for choice in modelchoice:
        if config.type == "DNA":
            models.append(DNA_model[choice])
        elif config.type == "RNA":
            models.append(RNA_model[choice])
        elif config.type == "prot":
            models.append(prot_model[choice])

    requests.post(requests_url, util_json.get_json(config.learn_name, 1))

    try:
        SL_train(config, models)

        resultdir = '/data/result/' + config.learn_name
        zip_file = zipfile.ZipFile(resultdir + 'zipdir' + '.zip', 'w')
        # 把zfile整个目录下所有内容，压缩为new.zip文件
        zip_file.write(resultdir, compress_type=zipfile.ZIP_DEFLATED)
        zip_file.close()

        pictureArray = []
        pictureArray.append("http://server.wei-group.net" + '/result/' + config.learn_name + "/statistics.png")
        pictureArray.append("http://server.wei-group.net" + '/result/' + config.learn_name + "/ROC_PRC.png")
        for model in models:
            pictureArray.append("http://server.wei-group.net" + '/result/' + config.learn_name + '/' + model + 'mer/t-sne.png')

        result = {
            "zip": "http://server.wei-group.net" + '/result/' + config.learn_name + 'zipdir' + '.zip',
            "pictures": pictureArray
        }

        postget = requests.post(requests_url, util_json.get_json(config.learn_name, 2, json.dumps(result)))
        print(postget.text)
    except Exception as e:
        print(e)
        requests.post(requests_url, util_json.get_json(config.learn_name, -1))
    # SL_train(config, kmerarray)

if __name__ == '__main__':
    # server_use()
    gpu_test()
