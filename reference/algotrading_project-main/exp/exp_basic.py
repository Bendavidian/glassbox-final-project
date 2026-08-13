import os
import torch
import numpy as np
import pandas as pd
import neptune
import time
from  utils.tools     import  EarlyStopping, adjust_learning_rate, visual, test_params_flop
from  utils.metrics   import metric
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix



class Exp_Basic(object):
    def __init__(self, args):
        
        self.args       = args
        self.device     = self._acquire_device()
        self.model      = self._build_model().to(self.device)

    
    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        pass

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass


    def print_update_inside_epochs(self,i,epoch,train_epochs,train_steps,loss,time_now,iter_count):
        if (i + 1) % 100 == 0:
            print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
            speed      = (time.time() - time_now) / iter_count
            left_time  = speed * ((train_epochs - epoch) * train_steps - i)
            iter_count = 0
            time_now   = time.time()
            print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))

    def paint_save_test(self,preds,trues,setting):
        mae, mse, rmse, mape, mspe, rse, corr,shr = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        #print(f'Sharp: {shr}')

    def plot_graph1(self,pred, true):
        fig, axs = plt.subplots(2, figsize=(15, 10))

        labels = ["SPY-O", "SPY-H", "SPY-L", "SPY-C","QQQ-O", "QQQ-H", "QQQ-L", "QQQ-C","DIA-O", "DIA-H", "DIA-L", "DIA-C", ] # Repeat the OHLC labels to match the data length

        # Plot predictions and true values
        for i in range(pred.shape[0]):
            axs[0].plot(pred[i], label=f'Pred {i+1}', marker='o')
            axs[1].plot(true[i], label=f'GT {i+1}', marker='x')
        
        axs[0].set_title('Predictions')
        axs[1].set_title('GroundTruth')
        axs[0].set_xlabel('Columns')
        axs[1].set_xlabel('Columns')
        axs[0].set_ylabel('Values')
        axs[1].set_ylabel('Values')
        axs[0].set_xticks(range(len(labels)))
        axs[1].set_xticks(range(len(labels)))
        axs[0].set_xticklabels(labels, rotation=45)
        axs[1].set_xticklabels(labels, rotation=45)
        axs[0].legend()
        axs[1].legend()
        axs[0].grid(True)
        axs[1].grid(True)

        plt.tight_layout()
        plt.savefig("plot1_custom_labels.png")
        plt.show()

    def plot_graph2(self,pred, true,low,up):
        fig, axs = plt.subplots(2, figsize=(15, 10))
        s  = [1 if pred[3][k] - pred[0][k] >= low and pred[3][k] - pred[0][k] <= up else 0 for k in range(12)]
        s2 = [1 if true[3][k] - true[0][k] >= low and true[3][k] - true[0][k] <= up else 0 for k in range(12)]

        labels = ["SPY-O", "SPY-H", "SPY-L", "SPY-C","QQQ-O", "QQQ-H", "QQQ-L", "QQQ-C","DIA-O", "DIA-H", "DIA-L", "DIA-C", ] # Repeat the OHLC labels to match the data length

        # Plot predictions and true values

        axs[0].plot(s, label=f'Pred-C', marker='o')
        axs[1].plot(s2, label=f'GT-C', marker='x')
        axs[0].set_title(f'{low}<= Predictions <= {up}')
        axs[1].set_title(f'{low}<= GroundTruth <= {up}')
        axs[0].set_xlabel('Columns')
        axs[1].set_xlabel('Columns')
        axs[0].set_ylabel('Values')
        axs[1].set_ylabel('Values')
        axs[0].set_xticks(range(len(labels)))
        axs[1].set_xticks(range(len(labels)))
        axs[0].set_xticklabels(labels, rotation=45)
        axs[1].set_xticklabels(labels, rotation=45)
        axs[0].legend()
        axs[1].legend()
        axs[0].grid(True)
        axs[1].grid(True)

        plt.tight_layout()
        plt.savefig("plot2_custom_labels.png")
        plt.show()

    def plot_graph3(self,pred, true, low, up):
        fig, axs = plt.subplots(2, figsize=(15, 10))
        s, s2 = [], []

        # Create the s and s2 lists
        s.append([low] * 12)
        s2.append([low] * 12)
        s.append([up] * 12)
        s2.append([up] * 12)
        s.append([pred[3][k] - pred[0][k] for k in range(12)])
        s2.append([true[3][k] - true[0][k] for k in range(12)])

        labels = ["SPY-O", "SPY-H", "SPY-L", "SPY-C", "QQQ-O", "QQQ-H", "QQQ-L", "QQQ-C", "DIA-O", "DIA-H", "DIA-L", "DIA-C"]

        # Plot predictions and true values
        axs[0].plot(s[0], label=f'LowerBound', marker='o')
        axs[1].plot(s2[0], label=f'LowerBound', marker='x')
        axs[0].plot(s[1], label=f'PRED-1-Diff', marker='o')
        axs[1].plot(s2[1], label=f'GT-1-Diff', marker='x')
        axs[0].plot(s[2], label=f'UpperBound', marker='o')
        axs[1].plot(s2[2], label=f'UpperBound', marker='x')

        axs[0].set_title('Predictions')
        axs[1].set_title('GroundTruth')
        axs[0].set_xlabel('Columns')
        axs[1].set_xlabel('Columns')
        axs[0].set_ylabel('Values')
        axs[1].set_ylabel('Values')
        axs[0].set_xticks(range(len(labels)))
        axs[1].set_xticks(range(len(labels)))
        axs[0].set_xticklabels(labels, rotation=45)
        axs[1].set_xticklabels(labels, rotation=45)
        axs[0].legend()
        axs[1].legend()
        axs[0].grid(True)
        axs[1].grid(True)

        plt.tight_layout()
        plt.savefig("plot3_custom_labels.png")
        plt.show()

    def plot_graph4(self,pred, true):
        fig, axs = plt.subplots(2, figsize=(15, 10))
        s, s2 = [], []

        # Create the s and s2 lists
        s.append([pred[k][3] for k in range(4)])
        s2.append([true[k][3] for k in range(4)])

        labels = ["SPY-C1", "SPY-C2","SPY-C3","SPY-C4"] 

        # Plot predictions and true values
        axs[0].plot(s[0], label=f'Value-Pred', marker='o')
        axs[1].plot(s2[0], label=f'value-GT', marker='x')

        axs[0].set_title('Predictions')
        axs[1].set_title('GroundTruth')
        axs[0].set_xlabel('Columns')
        axs[1].set_xlabel('Columns')
        axs[0].set_ylabel('Values')
        axs[1].set_ylabel('Values')
        axs[0].set_xticks(range(len(labels)))
        axs[1].set_xticks(range(len(labels)))
        axs[0].set_xticklabels(labels, rotation=45)
        axs[1].set_xticklabels(labels, rotation=45)
        axs[0].legend()
        axs[1].legend()
        axs[0].grid(True)
        axs[1].grid(True)

        plt.tight_layout()
        plt.savefig("plot4_custom_labels.png")
        plt.show()


    def plot_confuse_matrix(self,pred, true):
        cm = confusion_matrix(true, pred)
        # Plot the confusion matrix
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', cbar=False, 
                    xticklabels=['Predicted Negative', 'Predicted Positive'], 
                    yticklabels=['Actual Negative', 'Actual Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig("confuse.png")
        plt.show()





