import numpy as np
import pickle
from configuration import config_init
from frame import Learner

import os
import sys


def SL_train():
    config = config_init.get_config()
    learner = Learner.Learner(config)
    learner.setIO()
    learner.setVisualization()
    learner.load_data()
    learner.init_model()
    # learner.load_params()
    learner.init_optimizer()
    learner.def_loss_func()
    learner.train_model()
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
    SL_train()