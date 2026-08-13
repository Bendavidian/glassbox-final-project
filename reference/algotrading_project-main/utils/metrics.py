import numpy as np
import pandas as pd


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def SHARP2(pred, true):
    pred_flat = pred.flatten()
    true_flat = true.flatten()
    pred_series = pd.Series(pred_flat)
    true_series = pd.Series(true_flat)
    returns     = pred_series.pct_change().fillna(0) - true_series.pct_change().fillna(0)
    #df          = pred_series - true_series
    #returns     = df.pct_change().fillna(0)
    mean_returns = returns.mean()
    std_returns = returns.std()
    return  mean_returns / std_returns
    '''df      = pred - true
    returns = df.pct_change().fillna(0)
    return returns.mean()/returns.std()
    #return np.mean(np.abs(pred - true))/np.std(np.abs(pred - true))'''

def SHARP2(pred, true):
    #print(pred.shape)
    pred = true+0.01
    a =np.diff(pred)
    print(a)
    print(f'true sharp: {np.mean(np.abs(true))/np.std(np.abs(true))}')
    print(f'pred sharp: {np.mean(np.abs(pred))/np.std(np.abs(pred))}')
    return np.mean((pred - true) ** 2)/ np.std(true)



def SHARP(pred, true):

    a = np.abs(pred-true)
    b = np.sign(pred-true)
    c = b*a
    sharps1,sharps2,sharps3 = [],[],[]
    for i in range(a.shape[1]):
        sharps1.append(np.mean(c[:][i])/ np.std(true[:][i]))
        sharps2.append(np.mean(pred[:][i])/ np.std(pred[:][i]))
        sharps3.append(np.mean(true[:][i])/ np.std(true[:][i]))
    #print(sharps1)
    #print("----------true--------------")
    #print(sharps2)
    #print("----------pred--------------")
    #print(sharps3)

    

    #print(f'true sharp: {np.mean(np.abs(true))/np.std(np.abs(true))}')
    #print(f'pred sharp: {np.mean(np.abs(pred))/np.std(np.abs(pred))}')
    return 12
    #return np.mean(np.abs(sharps))

def SHARP5(pred, true):
    # Add 0.001 to all elements of the pred array
    #pred = true+0.1

    def calculate_sharp(pred, true):
        
        
        assert pred.shape == true.shape, "The shape of pred and true arrays must be the same."
        
        # Flatten the arrays if they are multi-dimensional
        if len(pred.shape) > 1:
            pred_flat = pred.flatten()
            true_flat = true.flatten()
        else:
            pred_flat = pred
            true_flat = true
        
        # Compute the difference
        df = pred_flat - true_flat
        
        # Calculate returns (percentage change) using NumPy
        returns = np.diff(df) / df[:-1]
        returns = np.insert(returns, 0, 0)  # Insert 0 return for the first period

        returns2 = np.diff(true_flat)/true_flat[:-1]
        returns2 = np.insert(returns2, 0, 0) 
        
        # Calculate mean and standard deviation of returns
        mean_returns = np.mean(returns)
        std_returns  = np.std(returns)
        
        # Calculate the Sharpe Ratio
        sharpe_ratio = mean_returns / std_returns
        
        return sharpe_ratio

    # Adjust for two-dimensional pred and true arrays
    assert len(pred.shape) == 2 and len(true.shape) == 2, "pred and true must be two-dimensional arrays"
    assert pred.shape == true.shape, "The shape of pred and true arrays must be the same."
    
    sharpe_ratios = np.zeros(pred.shape[1])

    for j in range(pred.shape[1]):
        sharpe_ratios[j] = calculate_sharp(pred[:, j], true[:, j])

    print("Sharpe Ratios for each time series:")
    print(sharpe_ratios)
    #print(len(sharpe_ratios))
    return np.mean(np.abs(sharpe_ratios))
    #return np.mean(sharpe_ratios)

def SHARP5(pred, true):
    def calculate_sharp(pred, true):
        # Ensure that pred and true have the same shape
        assert pred.shape == true.shape, "The shape of pred and true arrays must be the same."
        
        # Flatten the arrays if they are multi-dimensional
        if len(pred.shape) > 1:
            pred_flat = pred.flatten()
            true_flat = true.flatten()
        else:
            pred_flat = pred
            true_flat = true
        
        # Convert to pandas Series
        pred_series = pd.Series(pred_flat)
        true_series = pd.Series(true_flat)
        
        # Compute the signed difference
        df = pred_series - true_series
        
        # Calculate returns (percentage change)
        returns = df.pct_change().fillna(0)
        returns2 = true_series.pct_change().fillna(0)
        
        # Calculate mean and standard deviation of returns
        mean_returns = returns.mean()
        std_returns = returns2.std()
        
        # Calculate the Sharpe Ratio
        sharpe_ratio = mean_returns / std_returns
        
        return sharpe_ratio
    
    print(pred.shape)
    sharpe_ratios = np.zeros((pred.shape[1], pred.shape[2]))

    for i in range(pred.shape[1]):
        for j in range(pred.shape[2]):
            sharpe_ratios[i, j] = calculate_sharp(pred[:, i, j], true[:, i, j])

    print("Sharpe Ratios for each time series:")
    print(sharpe_ratios)
    return np.mean(sharpe_ratios)



def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    shr  = SHARP(pred,true)

    return mae, mse, rmse, mape, mspe, rse, corr,shr
