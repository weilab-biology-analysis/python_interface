import numpy as np
import pickle
import requests

from configuration import config_init
from frame import Learner
from util import util_plot, util_json

def SL_train(config):

    roc_datas, prc_datas = [], []
    name = []
    for index in [3, 4, 5, 6]:
        config.kmer = index
        learner = Learner.Learner(config)
        learner.setIO()
        learner.setVisualization()
        learner.load_data()
        learner.init_model()
        # learner.load_params()
        learner.init_optimizer()
        learner.def_loss_func()
        learner.train_model()

        roc_datas.append(learner.visualizer.roc_data)
        prc_datas.append(learner.visualizer.prc_data)
        name.append(str(index) + "mer")
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


if __name__ == '__main__':
    requests_url = "https://skyemperor.top/biology/job/status/update"
    config = config_init.get_config()
    requests.post(requests_url, util_json(config.learnname, 1))
    try:
        SL_train(config)
    except Exception as e:
        requests.post(requests_url, util_json(config.learnname, -1))
    requests.post(requests_url, util_json(config.learnname, 2))