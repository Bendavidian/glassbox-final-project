
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os


def get_vix_historical_data(data_path):
    root_path  = "./dataset/"
    df         = pd.read_csv(os.path.join(root_path,data_path))
    df_cleaned = df.dropna()
    data_multiplied = df_cleaned.copy()
    numeric_cols = data_multiplied.select_dtypes(include='number').columns
    data_multiplied[numeric_cols] = data_multiplied[numeric_cols] 
    return data_multiplied

def plot_clean_data(data_path):
    root_path  = "./dataset/"
    df         = pd.read_csv(os.path.join(root_path, data_path))
    df_cleaned = df.dropna()
    data_multiplied = df_cleaned.copy()
    numeric_cols = data_multiplied.select_dtypes(include='number').columns
    data_multiplied[numeric_cols] = data_multiplied[numeric_cols] 

    
    
    labels = ["date","SPY-Open", "SPY-High", "SPY-Low", "SPY-Close",
              "QQQ-Open", "QQQ-High", "QQQ-Low", "QQQ-Close",
              "DIA-Open", "DIA-High", "DIA-Low", "DIA-Close"]
    
    #label_index = 4
    plt.figure(figsize=(10, 6))
    plt.plot(data_multiplied.iloc[:, 4], label=labels[4])
    plt.title(labels[4])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot_spy.png")
    plt.show()
    
    '''fig, axs = plt.subplots(4, 3, figsize=(15, 10))
    
    axs = axs.flatten()
    
    for i in range(1,13):
        axs[i-1].plot(data_multiplied.iloc[:, i], label=labels[i])
        axs[i-1].set_title(labels[i])
        axs[i-1].legend()
        axs[i-1].grid(True)

    plt.tight_layout()
    plt.savefig("plot_features.png")
    plt.show()'''


def get_val_historical_data(data_path):
    root_path  = "./dataset/"
    df         = pd.read_csv(os.path.join(root_path,data_path))
    df_cleaned = df.dropna()
    data_multiplied = df_cleaned.copy()
    return data_multiplied
