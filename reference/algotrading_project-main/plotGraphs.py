from   models_backtest   import StrategySignal
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mpl_dates
from mplfinance.original_flavor import candlestick_ohlc

def plotGraphStrategy(df, title, path):
    print(df.shape)
    # Extracting Data for plotting
    ohlc = df.loc[:, ['date', 'Open', 'High', 'Low', 'Close']]

    # Converting date into datetime format
    ohlc['date'] = pd.to_datetime(ohlc['date'])
    ohlc['date'] = ohlc['date'].apply(mpl_dates.date2num)

    # Convert other columns to float for plotting
    ohlc[['Open', 'High', 'Low', 'Close']] = ohlc[['Open', 'High', 'Low', 'Close']].astype(float)

    # Create buy and sell signals (replace with your actual signals)
    print(df.shape)
    print(df['strategy_signal'])
    buy_signals = df[df['strategy_signal'] == StrategySignal.ENTER_LONG]
    sell_signals = df[df['strategy_signal'] == StrategySignal.CLOSE_LONG]

    # Converting signal dates
    buy_signals['date'] = pd.to_datetime(buy_signals['date']).apply(mpl_dates.date2num)
    sell_signals['date'] = pd.to_datetime(sell_signals['date']).apply(mpl_dates.date2num)

    # Creating Subplots
    fig, ax = plt.subplots()

    # Plot candlesticks
    candlestick_ohlc(ax, ohlc.values, width=0.6, colorup='green', colordown='red', alpha=0.8)

    # Plotting buy and sell signals
    ax.scatter(buy_signals['date'], buy_signals['Low'] - buy_signals['Low'] / 10, 
               marker='^', color='g', s=100, label='Buy Signal')
    ax.scatter(sell_signals['date'], sell_signals['High'] + sell_signals['High'] / 10, 
               marker='v', color='r', s=100, label='Sell Signal')

    # Setting labels & titles
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    fig.suptitle(title)

    # Formatting Date
    date_format = mpl_dates.DateFormatter('%d/%m/%Y')
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    fig.tight_layout()

    plt.legend()
    plt.savefig(path, dpi=300)
    plt.show()
    plt.close()

# Example usage
# df should be your DataFrame and StrategySignal should be defined appropriately
# plotGraphStrategy(df, "Sample Strategy", "output_path.png")
