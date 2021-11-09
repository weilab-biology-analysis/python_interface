import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from model import DNAbert, Protbert, Focal_Loss, TextCNN, TextRCNN, LSTMwithAttention, VDCNN, TransformerEncoder, GRU, LSTM, BiLSTM, DNN, RNN, ReformerEncoder, LinformerEncoder, PerformerEncoder, RoutingTransformerEncoder, RNN_CNN
from model import TextGCN
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
from copy import deepcopy
from util import util_transGraph

class ModelManager():
    def __init__(self, learner):
        self.learner = learner
        self.IOManager = learner.IOManager
        self.visualizer = learner.visualizer
        self.dataManager = learner.dataManager
        self.config = learner.config

        self.mode = self.config.mode
        self.model = None
        self.optimizer = None
        self.loss_func = None

        # supervised learning
        self.best_performance = None
        self.best_repres_list = None
        self.best_label_list = None
        self.test_performance = []
        self.valid_performance = []
        self.avg_test_loss = 0

        self.predict_answer = None

    def init_model(self):
        # Todo
        if self.mode == 'train-test':
            if self.config.model in ["3mer_DNAbert","4mer_DNAbert","5mer_DNAbert","6mer_DNAbert"]:
                self.model = DNAbert.BERT(self.config)
            elif self.config.model == 'prot_bert_bfd' or self.config.model == 'prot_bert':
                self.model = Protbert.BERT(self.config)
            elif self.config.model == "TransformerEncoder":
                self.model = TransformerEncoder.TransformerEncoder(self.config)
            elif self.config.model == "TextCNN":
                self.model = TextCNN.TextCNN(self.config)
            elif self.config.model == "LSTM":
                self.model = LSTM.LSTM(self.config)
            elif self.config.model == "DNN":
                self.model = DNN.DNN(self.config)
            elif self.config.model == "Reformer_Encoder":
                self.model = ReformerEncoder.ReformerEncoder(self.config)
            elif self.config.model == "Linformer_Encoder":
                self.model = LinformerEncoder.LinformerEncoder(self.config)
            elif self.config.model == "Performer_Encoder":
                self.model = PerformerEncoder.PerformerEncoder(self.config)
            elif self.config.model == "RoutingTransformer_Encoder":
                self.model = RoutingTransformerEncoder.RoutingTransformerEncoder(self.config)
            elif self.config.model == "VDCNN":
                self.model = VDCNN.VDCNN(self.config)
            elif self.config.model == "TextRCNN":
                self.model = TextRCNN.TextRCNN(self.config)
            elif self.config.model == "LSTMAttention":
                self.model = LSTMwithAttention.LSTMAttention(self.config)
            elif self.config.model == "BiLSTM":
                self.model = BiLSTM.BiLSTM(self.config)
            elif self.config.model == "RNN_CNN":
                self.model = RNN_CNN.RNN_CNN(self.config)
            elif self.config.model == "RNN":
                self.model = RNN.RNN(self.config)
            elif self.config.model == "GRU":
                self.model = GRU.GRU(self.config)
            elif self.config.model == "TextGCN":
                Graph = self.dataManager.get_dataloder(name='train_set')
                t_features, t_y_train, t_y_val, t_y_test, t_train_mask, tm_train_mask = Graph.get_titem()
                t_support, num_supports = Graph.get_suppport(0)
                self.model = TextGCN.GCN(t_features.shape[0], support=t_support)
            else:
                self.IOManager.log.Error('No Such Model')
            if self.config.model in ["TextGCN"]:
                pass
            elif self.config.cuda:
                self.model.cuda()
        else:
            self.IOManager.log.Error('No Such Mode')

    def load_params(self):
        if self.config.path_params:
            if self.mode == "train-test":
                self.model = self.__load_params(self.model, self.config.path_params)
            else:
                self.IOManager.log.Error('No Such Mode')
        else:
            self.IOManager.log.Warn('Path of Parameters Not Exist')

    def adjust_model(self):
        self.model = self.__adjust_model(self.model)

    def init_optimizer(self):
        if self.config.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.config.lr,
                                               weight_decay=self.config.reg)
        elif self.config.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.config.lr,
                                              weight_decay=self.config.reg)
        else:
            self.IOManager.log.Error('No Such Optimizer')

    def def_loss_func(self):
        if self.mode == 'train-test':
            if self.config.loss_func == 'CE':
                self.loss_func = nn.CrossEntropyLoss()
            elif self.config.loss_func == 'FL':
                if self.config.alpha != None:
                    alpha = torch.tensor([self.config.alpha, 1 - self.config.alpha])
                else:
                    alpha = None
                if self.config.cuda:
                    alpha.cuda()
                self.loss_func = Focal_Loss.FocalLoss(self.config.num_class, alpha=alpha,
                                                      gamma=self.config.gamma)
            else:
                self.IOManager.log.Error('No Such Loss Function')
        else:
            self.IOManager.log.Error('No Such Mode')

    def train(self):
        self.model.train()
        if self.mode == 'train-test':
            train_dataloader = self.dataManager.get_dataloder(name='train_set')
            test_dataloader = self.dataManager.get_dataloder(name='test_set')
            if self.config.model in ['TextGCN']:
                best_performance, best_repres_list, best_pos_list, best_neg_list, best_label_list, ROC, PRC = self.__SL_GNN_train(train_dataloader)
            else:
                best_performance, best_repres_list, best_pos_list, best_neg_list, best_label_list, ROC, PRC = self.__SL_train(train_dataloader, test_dataloader)
            self.visualizer.roc_data = ROC
            self.visualizer.prc_data = PRC
            self.visualizer.repres_list = best_repres_list
            self.visualizer.label_list = best_label_list
            self.visualizer.pos_list = best_pos_list
            self.visualizer.neg_list = best_neg_list

            self.best_repres_list = best_repres_list
            self.best_label_list = best_label_list
            self.best_performance = best_performance
            self.IOManager.log.Info('Best Performance: {}'.format(self.best_performance))
            self.IOManager.log.Info('Performance: {}'.format(self.test_performance))
        elif self.mode == 'cross validation':
            for k in range(self.config.k_fold):
                train_dataloader = self.dataManager.get_dataloder(name='train_set')[k]
                valid_dataloader = self.dataManager.get_dataloder(name='valid_set')[k]
                best_valid_performance = self.__cross_validation(train_dataloader, valid_dataloader)
                self.valid_performance.append(best_valid_performance)
            self.IOManager.log.Info('valid_performance: ', self.valid_performance)
        else:
            self.IOManager.log.Error('No Such Mode')

    def test(self):
        self.model.eval()
        if self.mode == 'train-test' or self.mode == 'cross validation':
            test_dataloader = self.dataManager.get_dataloder('test_set')
            if test_dataloader is not None:
                test_performance, avg_test_loss = self.__SL_test(test_dataloader)
                log_text = '\n' + '=' * 20 + ' Final Test Performance ' + '=' * 20 \
                           + '\n[ACC,\tSE,\t\tSP,\t\tAUC,\tMCC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
                    test_performance[0], test_performance[1], test_performance[2], test_performance[3],
                    test_performance[4]) \
                           + '\n' + '=' * 60
                self.IOManager.log.Info(log_text)
            else:
                self.IOManager.log.Warn('Test Data is None.')
        elif self.mode == 'annotation':
            predict_data = self.dataManager.predict_data
            predict_num = self.dataManager.predict_num
            self.predict_answer = self.__predict(predict_data, predict_num)
        else:
            self.IOManager.log.Error('No Such Mode')

    def __load_params(self, model, param_path):
        pretrained_dict = torch.load(param_path)
        new_model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
        new_model_dict.update(pretrained_dict)
        model.load_state_dict(new_model_dict)
        return model

    def __adjust_model(self, model):
        print('-' * 50, 'Model.named_parameters', '-' * 50)
        for name, value in model.named_parameters():
            print('[{}]->[{}],[requires_grad:{}]'.format(name, value.shape, value.requires_grad))

        # Count the total parameters
        params = list(model.parameters())
        k = 0
        for i in params:
            l = 1
            for j in i.size():
                l *= j
            k = k + l
        print('=' * 50, "Number of total parameters:" + str(k), '=' * 50)
        return model

    def __get_loss(self, logits, label):
        loss = 0
        if self.config.loss_func == 'CE':
            loss = self.loss_func(logits.view(-1, self.config.num_class), label.view(-1))
            loss = (loss.float()).mean()
            # flooding method
            loss = (loss - 0.02).abs() + 0.02
            # loss = (loss - self.config.b).abs() + self.config.b
        elif self.config.loss_func == 'FL':
            # logits = logits.view(-1, 2)
            # label = label.view(-1)
            loss = self.loss_func(logits, label)
        return loss

    def __caculate_metric(self, pred_prob, label_pred, label_real):
        test_num = len(label_real)
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for index in range(test_num):
            if label_real[index] == 1:
                if label_real[index] == label_pred[index]:
                    tp = tp + 1
                else:
                    fn = fn + 1
            else:
                if label_real[index] == label_pred[index]:
                    tn = tn + 1
                else:
                    fp = fp + 1

        # print('tp\tfp\ttn\tfn')
        # print('{}\t{}\t{}\t{}'.format(tp, fp, tn, fn))

        # Accuracy
        ACC = float(tp + tn) / test_num

        # Precision
        # if tp + fp == 0:
        #     Precision = 0
        # else:
        #     Precision = float(tp) / (tp + fp)

        # Sensitivity
        if tp + fn == 0:
            Recall = Sensitivity = 0
        else:
            Recall = Sensitivity = float(tp) / (tp + fn)

        # Specificity
        if tn + fp == 0:
            Specificity = 0
        else:
            Specificity = float(tn) / (tn + fp)

        # MCC
        if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
            MCC = 0
        else:
            MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

        # F1-score
        # if Recall + Precision == 0:
        #     F1 = 0
        # else:
        #     F1 = 2 * Recall * Precision / (Recall + Precision)

        # ROC and AUC
        FPR, TPR, thresholds = roc_curve(label_real, pred_prob, pos_label=1)  # Default 1 is positive sample
        AUC = auc(FPR, TPR)

        # PRC and AP
        precision, recall, thresholds = precision_recall_curve(label_real, pred_prob, pos_label=1)
        AP = average_precision_score(label_real, pred_prob, average='macro', pos_label=1, sample_weight=None)

        # ROC(FPR, TPR, AUC)
        # PRC(Recall, Precision, AP)

        performance = [ACC, Sensitivity, Specificity, AUC, MCC]
        roc_data = [FPR, TPR, AUC]
        prc_data = [recall, precision, AP]
        return performance, roc_data, prc_data

    def __cross_validation(self, train_dataloader, valid_dataloader):
        step = 0
        best_mcc = 0
        best_valid_performance = 0

        for epoch in range(1, self.config.epoch + 1):
            for batch in train_dataloader:
                data, label = batch
                logits, _ = self.model(data)
                train_loss = self.__get_loss(logits, label)

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                step += 1

                '''Periodic Train Log'''
                if step % self.config.interval_log == 0:
                    corrects = (torch.max(logits, 1)[1] == label).sum()
                    the_batch_size = label.shape[0]
                    train_acc = 100.0 * corrects / the_batch_size
                    print('Epoch[{}] Batch[{}] - loss: {:.6f} | ACC: {:.4f}%({}/{})'.format(epoch, step, train_loss,
                                                                                            train_acc,
                                                                                            corrects,
                                                                                            the_batch_size))
                    self.visualizer.step_log_interval.append(step)
                    self.visualizer.train_metric_record.append(train_acc)
                    self.visualizer.train_loss_record.append(train_loss)

            '''Periodic Valid'''
            if epoch % self.config.interval_valid == 0:
                valid_performance, avg_test_loss = self.__SL_test(valid_dataloader)
                self.visualizer.step_test_interval.append(epoch)
                self.visualizer.test_metric_record.append(valid_performance[0])
                self.visualizer.test_loss_record.append(avg_test_loss)
                valid_mcc = valid_performance[4]  # test_performance: [ACC, Sensitivity, Specificity, AUC, MCC]
                if valid_mcc > best_mcc:
                    best_mcc = valid_mcc
                    best_valid_performance = valid_performance
                    if self.config.save_best :
                        self.IOManager.save_model_dict(self.model.state_dict(), self.config.model_save_name,
                                                       'MCC', best_mcc)
        return best_valid_performance

    def __SL_train(self, train_dataloader, test_dataloader):
        step = 0
        best_mcc = 0
        best_performance = None
        best_ROC = None
        best_PRC = None
        best_repres_list = None
        best_label_list = None
        best_pos_list = None
        best_neg_list = None

        for epoch in range(1, self.config.epoch + 1):
            self.model.train()
            for batch in train_dataloader:
                data, label = batch
                # print(label)
                logits, _ = self.model(data)
                train_loss = self.__get_loss(logits, label)

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                step += 1

                '''Periodic Train Log'''
                if step % self.config.interval_log == 0:
                    corrects = (torch.max(logits, 1)[1] == label).sum()
                    the_batch_size = label.shape[0]
                    train_acc = 100.0 * corrects / the_batch_size
                    print('Epoch[{}] Batch[{}] - loss: {:.6f} | ACC: {:.4f}%({}/{})'.format(epoch, step, train_loss,
                                                                                            train_acc,
                                                                                            corrects,
                                                                                            the_batch_size))
                    self.visualizer.step_log_interval.append(step)

                    if self.config.cuda:
                        self.visualizer.train_metric_record.append(train_acc.cpu().detach().numpy())
                        self.visualizer.train_loss_record.append(train_loss.cpu().detach().numpy())
                    else:
                        self.visualizer.train_metric_record.append(train_acc.cpu().detach().numpy())
                        self.visualizer.train_loss_record.append(train_loss.cpu().detach().numpy())

            '''Periodic Test'''
            if epoch % self.config.interval_test == 0:
                test_performance, avg_test_loss, ROC_data, PRC_data, repres_list, label_list, pos_list, neg_list = self.__SL_test(test_dataloader)
                self.visualizer.step_test_interval.append(epoch)
                self.visualizer.test_metric_record.append(test_performance[0])
                self.visualizer.test_loss_record.append(avg_test_loss.cpu().detach().numpy())
                self.test_performance.append(test_performance)

                log_text = '\n' + '=' * 20 + ' Test Performance. Epoch[{}] '.format(epoch) + '=' * 20 \
                           + '\n[ACC,\tSE,\t\tSP,\t\tAUC,\tMCC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
                    test_performance[0], test_performance[1], test_performance[2], test_performance[3],
                    test_performance[4]) \
                           + '\n' + '=' * 60
                self.IOManager.log.Info(log_text)

                test_mcc = test_performance[0]  # test_performance: [ACC, Sensitivity, Specificity, AUC, MCC]
                if test_mcc >= best_mcc:
                    best_mcc = test_mcc
                    best_performance = test_performance
                    best_ROC = ROC_data
                    best_PRC = PRC_data
                    best_repres_list = repres_list
                    best_label_list = label_list
                    best_pos_list = pos_list
                    best_neg_list = neg_list
                    if self.config.save_best :
                        self.IOManager.save_model_dict(self.model.state_dict(), self.config.model_save_name,
                                                       'MCC', best_mcc)
        return best_performance, best_repres_list, best_pos_list, best_neg_list, best_label_list, best_ROC, best_PRC

    def __SL_test(self, dataloader):
        corrects = 0
        test_batch_num = 0
        test_sample_num = 0
        avg_test_loss = 0
        pred_prob = []
        label_pred = []
        label_real = []

        repres_list = []
        logits_list = []
        label_list = []
        pos_list = []
        neg_list = []

        self.model.eval()

        with torch.no_grad():
            for batch in dataloader:
                data, label = batch
                logits, representation = self.model(data)
                avg_test_loss += self.__get_loss(logits, label)

                repres_list.extend(representation.cpu().detach().numpy().tolist())
                logits_list.extend(logits.cpu().detach().numpy())
                label_list.extend(label.cpu().detach().numpy())

                pred_prob_all = F.softmax(logits, dim=1)  # 预测概率 [batch_size, class_num]
                pred_prob_positive = pred_prob_all[:, 1]  # 注意，极其容易出错
                pred_prob_negtive = pred_prob_all[:, 0]
                pred_prob_sort = torch.max(pred_prob_all, 1)  # 每个样本中预测的最大的概率 [batch_size]
                pred_class = pred_prob_sort[1]  # 每个样本中预测的最大的概率所在的位置（类别） [batch_size]

                pos_list.extend(pred_prob_positive.cpu().detach().numpy().tolist())
                neg_list.extend(pred_prob_negtive.cpu().detach().numpy().tolist())

                corrects += (pred_class == label).sum()
                test_sample_num += len(label)
                test_batch_num += 1
                pred_prob = pred_prob + pred_prob_positive.tolist()
                label_pred = label_pred + pred_class.tolist()
                label_real = label_real + label.tolist()

        performance, ROC_data, PRC_data = self.__caculate_metric(pred_prob, label_pred, label_real)

        avg_test_loss /= test_batch_num
        avg_acc = 100.0 * corrects / test_sample_num
        print('Evaluation - loss: {:.6f}  ACC: {:.4f}%({}/{})'.format(avg_test_loss,
                                                                      avg_acc,
                                                                      corrects,
                                                                      test_sample_num))

        self.avg_test_loss = avg_test_loss
        return performance, avg_test_loss, ROC_data, PRC_data, repres_list, label_list, pos_list, neg_list

    def __SL_GNN_train(self, train_dataloader):
        '''

        Args:
            path:  dataset file

        Returns:

        '''
        best_mcc = 0
        best_performance = None
        best_ROC = None
        best_PRC = None
        best_repres_list = None
        best_label_list = None
        best_logits_list = None
        best_pos_list = None
        best_neg_list = None
        # Graph = util_transGraph.CreateTextGCNGraph(self.config.path_data)
        # adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = Graph.load_corpus()
        # t_features, t_y_train, t_y_val, t_y_test, t_train_mask, tm_train_mask = \
        #     Graph.post_process(adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size,
        #                        test_size)
        Graph = train_dataloader
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = Graph.get_item()
        t_features, t_y_train, t_y_val, t_y_test, t_train_mask, tm_train_mask = Graph.get_titem()
        t_support, num_supports = Graph.get_suppport(0)
        # if torch.cuda.is_available():
        #     t_features = t_features.cuda()
        #     t_y_train = t_y_train.cuda()
        #     t_y_val = t_y_val.cuda()
        #     t_y_test = t_y_test.cuda()
        #     t_train_mask = t_train_mask.cuda()
        #     tm_train_mask = tm_train_mask.cuda()
        # (t_feature, _, _, _, _, _) = self.dataManager.get_dataloder(name='train_set')
        # self.model = TextGCN.GCN(t_features.shape[0], support=t_support, num_classes=2)
        for epoch in range(self.config.epoch):
            self.model.train()
            logits,_ = self.model(t_features)
            train_loss = self.__get_loss(logits * tm_train_mask, torch.max(t_y_train, 1)[1])
            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()

            if epoch % self.config.interval_test == 0:
                # prob, test_loss, test_acc, pred, labels, test_duration = evaluate(t_features, t_y_test, test_mask)
                test_performance, avg_test_loss, ROC_data, PRC_data, repres_list, label_list, pos_list, neg_list = self.__SL_GNN_test(t_features, t_y_test, test_mask)

                self.visualizer.step_test_interval.append(epoch)
                self.visualizer.test_metric_record.append(test_performance[0])
                self.visualizer.test_loss_record.append(avg_test_loss.cpu().detach().numpy())
                self.test_performance.append(test_performance)

                log_text = '\n' + '=' * 20 + ' Test Performance. Epoch[{}] '.format(epoch) + '=' * 20 \
                           + '\n[ACC,\tSE,\t\tSP,\t\tAUC,\tMCC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
                    test_performance[0], test_performance[1], test_performance[2], test_performance[3],
                    test_performance[4]) \
                           + '\n' + '=' * 60
                self.IOManager.log.Info(log_text)

                test_mcc = test_performance[0]  # test_performance: [ACC, Sensitivity, Specificity, AUC, MCC]
                if test_mcc >=  best_mcc:
                    best_mcc = test_mcc
                    best_performance = test_performance
                    best_ROC = ROC_data
                    best_PRC = PRC_data
                    best_repres_list = repres_list
                    best_label_list = label_list
                    best_pos_list = pos_list
                    best_neg_list = neg_list
                    if self.config.save_best:
                        self.IOManager.save_model_dict(self.model.state_dict(), self.config.model_save_name,
                                                       'MCC', best_mcc)
        return best_performance, best_repres_list, best_pos_list, best_neg_list, best_label_list, best_ROC, best_PRC

    def __SL_GNN_test(self, features, labels, mask):
        corrects = 0
        test_batch_num = 0
        test_sample_num = 0
        avg_test_loss = 0

        pred_prob = []

        label_pred = []
        label_real = []

        pred_prob_positive = []
        pred_prob_negtive = []

        repres_list = []
        label_list = []
        logits_list = []

        pos_list = []
        neg_list = []

        self.model.eval()

        with torch.no_grad():
            logits, representation = self.model(features)
            prob = torch.softmax(logits, dim=1)

            repres_list.extend(representation.cpu().detach().numpy().tolist())
            pred_prob_positive = prob[:, 1]  # 注意，极其容易出错
            pred_prob_negtive = prob[:, 0]
            pos_list.extend(pred_prob_positive.cpu().detach().numpy().tolist())
            neg_list.extend(pred_prob_negtive.cpu().detach().numpy().tolist())

            t_mask = torch.from_numpy(np.array(mask * 1., dtype=np.float32))
            tm_mask = torch.transpose(torch.unsqueeze(t_mask, 0), 1, 0).repeat(1, labels.shape[1])
            avg_test_loss = self.loss_func(logits * tm_mask, torch.max(labels, 1)[1])

            label_pred = torch.max(logits, 1)[1]
            all = int(torch.sum(t_mask).data)
            len_mask= t_mask.shape[0]
            pred_prob = prob[:,1][len_mask-all:]
            label_pred = label_pred[len_mask-all:]
            label_real = torch.max(labels, 1)[1][len_mask-all:]


        # print(all)
        # print(len_mask)
        # print(prob[len_mask-all:])
        # print('-------------------------------------------')
        # print((label_pred == torch.max(labels, 1)[1]).float())
        # print(t_mask)
        # print('-------------------------------')
        # print(((label_pred == torch.max(labels, 1)[1]).float()* t_mask))
        # print('-------------------------------------')
        # print(pred_prob)
        # print(label_pred)
        # print(label_real)
        performance, ROC_data, PRC_data = self.__caculate_metric(pred_prob, label_pred, label_real)

        return performance, avg_test_loss, ROC_data, PRC_data, repres_list, label_list, pos_list, neg_list

    def __predict(self, predict_data, predict_num):
        # predict_num 是以要预测的seq数量
        logits_data_all = []

        self.model.eval()
        # for seq in predict_data:
        logits, _ = self.model(predict_data)

        pred_prob_all = F.softmax(logits, dim=1)  # 预测概率 [batch_size, class_num]

        pred_prob_sort = torch.max(pred_prob_all, 1)  # 每个样本中预测的最大的概率 [batch_size]
        pred_class = pred_prob_sort[1]  # 每个样本中预测的最大的概率所在的位置（类别） [batch_size]

        count_sum = 0
        for index, count in enumerate(predict_num):
            for i in range(count):
                if pred_class[i + count_sum] == 1:
                    # 第几个序列， 在这条序列的第几个位置上， 置信度
                    logits_data_all.append([index, i, pred_prob_all[i + count_sum][1]])
            count_sum = +count

        return logits_data_all

