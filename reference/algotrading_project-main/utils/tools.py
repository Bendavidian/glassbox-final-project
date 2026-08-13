import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import pandas as pd
from  scipy.stats.mstats import winsorize

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def is_buy_signal(df,conf,low,up):
    #for i in range(1,4):
        #if calculate_confidence_signals_k(df,i)
    s = sum([1 if df[3][k] - df[0][k] >= low and df[3][k] - df[0][k] <= up else 0 for k in range(12)])
    return s>=conf and calculate_confidence_signals(df) >= conf

    #return calculate_confidence_signals(df) >= conf and df[3][3] - df[0][3] >= low and df[3][3] - df[0][3] <= up

def is_buy_signal_test(df,conf,low,up):
    bound = -1
    for i in range(1,4):
        if calculate_confidence_signals(df) >= conf and df[i][3] - df[0][3] >= low and df[i][3] - df[0][3] <= up:
            if i > bound:
                bound = i
    return bound


def which_i_to_buy(df):
    max_i = 3
    max   = -1
    for i in range(1,4):
        if df[i][3] > max :
            max  = df[i][3]
            max_i= i
    return max_i


def calculate_confidence_signals_k(df,j,conf):
    return sum([1 if df[j][k]>df[0][k] else 0 for k in range(12)]) >= conf



def calculate_confidence_signals(df):
    return sum([1 if df[3][k]>df[0][k] else 0 for k in range(12)])


def is_short_signal(df,conf,low,up):
    return calculate_confidence_signals_short(df) >= conf and df[0][3] - df[3][3] > low and df[0][3]-df[3][3] < up

def calculate_confidence_signals_short(df):
    return sum([1 if df[3][k]<df[0][k] else 0 for k in range(12)])

def winsorize_df(df):
    def winsorize_series(series, limits):
        return winsorize(series, limits=limits)

    limits = (0.01, 0.008)
    df_winsorized = df.apply(winsorize_series, limits=limits)
    df_winsorized = pd.DataFrame(df_winsorized, columns=df.columns)
    return df_winsorized


def calculate_sharpe_ratio(portfolio_values, risk_free_rate=0):
    # Calculate the periodic returns
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    # Calculate the average return
    avg_return = np.mean(returns)
    
    # Calculate the standard deviation of returns
    std_dev = np.std(returns)
    
    # Calculate the Sharpe Ratio
    sharpe_ratio = (avg_return - risk_free_rate) / std_dev
    
    return sharpe_ratio