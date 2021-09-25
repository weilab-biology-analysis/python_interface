import sys
import os
import pickle
import json
import zipfile

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import requests
from configuration import config_init
from frame import Learner
from util import util_plot, util_json


def SL_train(config, kmerarray):
    roc_datas, prc_datas = [], []
    name = []

    for index, kmer in enumerate(kmerarray):
        config.kmer = kmer
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
            print("statistics plot")
            util_plot.draw_statistics_bar(learner.dataManager.train_dataset, learner.dataManager.test_dataset, config)

        roc_datas.append(learner.visualizer.roc_data)
        prc_datas.append(learner.visualizer.prc_data)
        name.append(str(kmer) + "mer")
    util_plot.draw_ROC_PRC_curve(roc_datas, prc_datas, name, config)
    # learner.test_model()


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
    SL_train(config)


def server_use():
    os.chdir("/root/biology_python/main")
    # print(sys.argv[2])
    # print(type(sys.argv[2]))
    setting = json.loads(sys.argv[2])
    requests_url = "http://server.wei-group.net/biology/job/status/update"

    config = config_init.get_config()
    config.learn_name = str(setting["jobId"])
    config.path_data = setting["requestDataPath"]
    config.path_save = setting["resultDataPath"]

    if config.model == "DNAbert":
        kmerarray = [3, 4, 5, 6]
    else:
        kmerarray = [3]

    requests.post(requests_url, util_json.get_json(config.learn_name, 1))

    try:
        SL_train(config, kmerarray)

        resultdir = '/data/result/' + config.learn_name
        zip_file = zipfile.ZipFile(resultdir + 'zipdir' + '.zip', 'w')
        # 把zfile整个目录下所有内容，压缩为new.zip文件
        zip_file.write(resultdir, compress_type=zipfile.ZIP_DEFLATED)
        zip_file.close()

        picturearry = []
        picturearry.append(resultdir + "/statistics.jpeg")
        picturearry.append(resultdir + "/ROC_PRC.jpeg")
        for kmer in kmerarray:
            picturearry.append(resultdir + '/' + kmer + 'mer/t-sne.jpeg')

        result = {
            "zip": "http://server.wei-group.net" + resultdir + 'zipdir' + '.zip',
            "picture": picturearry
        }

        postget = requests.post(requests_url, util_json.get_json(config.learn_name, 2, json.dumps(result)))
        print(postget.text)
    except Exception as e:
        requests.post(requests_url, util_json.get_json(config.learn_name, -1))


if __name__ == '__main__':
    server_use()
    # gpu_test()
