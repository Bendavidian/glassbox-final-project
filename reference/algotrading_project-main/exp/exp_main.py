import numpy             as np
import pandas            as pd
import torch.nn          as nn
import torch
import matplotlib.pyplot as plt
import neptune
import warnings
import os
import time
import random
from data_provider.data_factory import  data_provider
from exp.exp_basic              import  Exp_Basic
from models                     import  DLinear, Linear, NLinear,Lstm
from utils.tools                import  EarlyStopping, adjust_learning_rate, visual, test_params_flop,is_buy_signal,is_short_signal,which_i_to_buy,is_buy_signal_test
from utils.metrics              import  metric
from torch                      import optim
from models_backtest import ActionType, PositionType, Position, StrategySignal




import sys

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        

    def _build_model(self):
        model_dict = {'DLinear': DLinear,'NLinear': NLinear,'Linear': Linear, 'Lstm': Lstm}
        model      = model_dict[self.args.model].Model(self.args).float()
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self, setting,val=False):
        if val:
            train_data, train_loader = self._get_data(flag='val')
            print("raz")
        else:
            train_data, train_loader = self._get_data(flag='train')

        path                     = os.path.join(self.args.checkpoints, setting)
        time_now                 = time.time()
        train_steps              = len(train_loader)
        early_stopping           = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim              = self._select_optimizer()
        criterion                = self._select_criterion()

        if not os.path.exists(path): os.makedirs(path)
                
        for epoch in range(10):
            print(f'epoch: {epoch}')
            iter_count = 0
            train_loss = []
            epoch_time = time.time()
            self.model.train()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad()
                iter_count += 1
                batch_x     = batch_x.float().to(self.device)
                batch_y     = batch_y.float().to(self.device)
                outputs     = self.model(batch_x)

                f_dim       = -1 if self.args.features == 'MS' else 0
                outputs     = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y     = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss        = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()

            train_loss = np.average(train_loss)
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(epoch + 1, train_steps, train_loss))
            early_stopping(train_loss, self.model, path)
            if early_stopping.early_stop: print("Early stopping"); break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        print(f'loss: {train_loss}')
        return self.model


    def test(self, setting,data,arg_conf,arg_low,arg_up,test=0):
        test_data, test_loader = self._get_data(flag='test')
        criterion              = self._select_criterion()
        model_optim            = self._select_optimizer()
        preds                  = []
        trues                  = []
        signal_pred            = []
        signal_gt              = []
        random.seed(1)

        self.model.eval()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            model_optim.zero_grad()

            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            outputs = self.model(batch_x)

            f_dim   = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
            loss    = criterion(outputs, batch_y)
            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()

            pred    = test_data.inverse_transform(outputs[0])
            true    = test_data.inverse_transform(batch_y[0])

            if i%4 ==0:
                loss.backward()
                model_optim.step()
                k     = 3
                if is_buy_signal(pred,arg_conf,arg_low,arg_up):
                    if i ==8:
                        self.plot_graph1(pred,true)
                        self.plot_graph2(pred,true,arg_low,arg_up)
                        self.plot_graph3(pred,true,arg_low,arg_up)
                        self.plot_graph4(pred,true)
                    k = which_i_to_buy(pred)
                    data['strategy_signal'][i]   = StrategySignal.ENTER_LONG
                    data['strategy_signal'][i+k] = StrategySignal.CLOSE_LONG
                    signal_pred.append(1)
                else:
                    signal_pred.append(0)

                if true[3][k] - true[0][k] >= arg_low:
                    signal_gt.append(1)
                else:
                    signal_gt.append(0)


            preds.append(outputs[0])
            trues.append(batch_y[0])

        preds          = np.concatenate(preds, axis=0)
        trues          = np.concatenate(trues, axis=0)
        self.plot_confuse_matrix(signal_pred,signal_gt)
        signal_acc     = np.mean(np.array(signal_pred) == np.array(signal_gt))
        false_sign     = np.sum([1 for i, (a, p) in enumerate(zip(signal_pred, signal_gt)) if a == 1 and p == 0])
        hit_acc        = np.sum([1 for i, (a, p) in enumerate(zip(signal_pred, signal_gt)) if a == 1 and p == 1])


        self.paint_save_test(preds,trues,setting)
        print(f'Signal accuracy: {signal_acc}')
        print(f'False signals  : {false_sign}')
        print(f'Hits Buy       : {hit_acc}')
        return

    def validate(self, setting,hyper_par):
        random.seed(1)
        confidence = hyper_par['confidence']
        lowerbound = hyper_par['lowerbound']
        upperbound = hyper_par['upperbound']
        mean_conf  = {conf:0 for conf in confidence}
        mean_low   = {low:0 for low  in lowerbound}
        mean_up    = {up:0 for up in upperbound}
        
        for conf in confidence:
            for low in lowerbound:
                for up in upperbound:
                    signal_pred              = []
                    signal_gt                = []
                    vali_data, vali_loader = self._get_data(flag='val')
                    self.model.eval()
                    with torch.no_grad():
                        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                            batch_x = batch_x.float().to(self.device)
                            batch_y = batch_y.float().to(self.device)
                            outputs = self.model(batch_x)

                            f_dim   = -1 if self.args.features == 'MS' else 0
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            outputs = outputs.detach().cpu().numpy()
                            batch_y = batch_y.detach().cpu().numpy()

                            pred    = vali_data.inverse_transform(outputs[0])
                            true    = vali_data.inverse_transform(batch_y[0])

                            if i%4==0:            
                                if is_buy_signal(pred,conf,low,up):
                                    signal_pred.append(1)
                                else:
                                    signal_pred.append(0)

                                if true[3][3] - true[0][3] > 0:
                                    signal_gt.append(1)
                                else:
                                    signal_gt.append(0)

                    signal_acc         = np.mean(np.array(signal_pred) == np.array(signal_gt))
                    mean_conf[conf]   +=signal_acc
                    mean_low[low]     +=signal_acc
                    mean_up[up]       +=signal_acc
                    
        print(f'ArgConfidence: {max(mean_conf, key=mean_conf.get)}')
        print(f'ArgLow       : {max(mean_low,  key=mean_low.get)}')
        print(f'ArgUp        : {max(mean_up,   key=mean_up.get)}')

        return max(mean_conf, key=mean_conf.get),max(mean_low,  key=mean_low.get),max(mean_up,   key=mean_up.get)

