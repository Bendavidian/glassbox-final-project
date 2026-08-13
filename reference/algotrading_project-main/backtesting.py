import pandas as pd
from datetime import datetime
import numpy as np
from models_backtest import ActionType, PositionType, Position, StrategySignal
from strategies      import BaseStrategy
from evaluation      import evaluate_strategy

    
def calc_realistic_price(row: pd.Series ,action_type: ActionType, slippage_factor=np.inf):
    slippage_rate = ((row['Close'] - row['Open']) / row['Open']) / slippage_factor
    slippage_price = row['Open'] + row['Open'] * slippage_rate
    
    if action_type == ActionType.BUY:
        return max(slippage_price, row['Open'])
    else:
        return min(slippage_price, row['Open'])   

def backtest(data: pd.DataFrame, strategy: BaseStrategy, starting_balance: int, slippage_factor: float=5.0, commission: float=0.0) -> pd.DataFrame:       
    
    def enter_position(data: pd.DataFrame, index: int, row: pd.Series, curr_qty: float, curr_balance: float, position_type: PositionType) -> Position:
        if position_type == PositionType.LONG:
            buy_price = calc_realistic_price(row, ActionType.BUY, slippage_factor=slippage_factor)
            qty_to_buy = int(strategy.calc_qty(buy_price, curr_balance, ActionType.BUY))
            position = Position(qty_to_buy, buy_price, position_type)
            data.loc[index, 'qty'] = curr_qty + qty_to_buy
            data.loc[index, 'balance'] = curr_balance - qty_to_buy * buy_price - commission
        
        elif position_type == PositionType.SHORT:
            sell_price = calc_realistic_price(row, ActionType.SELL, slippage_factor=slippage_factor)
            qty_to_sell = strategy.calc_qty(sell_price, curr_balance, ActionType.SELL)
            position = Position(qty_to_sell, sell_price, position_type)
            data.loc[index, 'qty'] = curr_qty - qty_to_sell
            data.loc[index, 'balance'] = curr_balance + qty_to_sell * sell_price - commission
        
        return position
    
    def close_position(data: pd.DataFrame, index: int, row: pd.Series, curr_qty: float, curr_balance: float, position: Position):
        if position.type == PositionType.LONG:
            sell_price = int(calc_realistic_price(row, ActionType.SELL, slippage_factor=slippage_factor))
            data.loc[index, 'qty'] = curr_qty - position.qty
            data.loc[index, 'balance'] = curr_balance + position.qty * sell_price - commission

        elif position.type == PositionType.SHORT:
            buy_price = calc_realistic_price(row, ActionType.BUY, slippage_factor=slippage_factor)
            data.loc[index, 'qty'] = curr_qty + position.qty
            data.loc[index, 'balance'] = curr_balance - position.qty * buy_price - commission
        
    
    # initialize df 
    data['qty'] = 0.0
    data['balance'] = 0.0

    # Calculate strategy signal
    strategy.calc_signal(data)
    
    # Loop through the data to calculate portfolio value
    position: Position = None
    data.reset_index(inplace=True)
    num_trading_days = data.shape[0]
    
    for index, row in data.iterrows():
        curr_qty = data.loc[index - 1, 'qty'] if index > 0 else 0
        curr_balance = data.loc[index - 1, 'balance'] if index > 0 else starting_balance
        
        # handle stop loss and take profit
        if position is not None:
            sl_tp_res = strategy.check_sl_tp(data.iloc[index - 1], position)
            if sl_tp_res is not None:
                sl_tp_qty, sl_tp_price, sl_tp_action = sl_tp_res
                if sl_tp_action == ActionType.BUY:
                    curr_balance = curr_balance - sl_tp_qty * sl_tp_price - commission
                    curr_qty = curr_qty + sl_tp_qty
                    
                elif sl_tp_action == ActionType.SELL:
                    curr_balance = curr_balance + sl_tp_qty * sl_tp_price - commission
                    curr_qty = curr_qty - sl_tp_qty
        
        # Close position at end of trade
        if index + 1 == num_trading_days and position is not None: 
            close_position(data, index, row, curr_qty, curr_balance, position)
                
        # Handle enter long signal
        elif row['strategy_signal'] == StrategySignal.ENTER_LONG:
            position = enter_position(data, index, row, curr_qty, curr_balance, PositionType.LONG)
        
        # Handle enter short signal  
        elif row['strategy_signal'] == StrategySignal.ENTER_SHORT: 
            position = enter_position(data, index, row, curr_qty, curr_balance, PositionType.SHORT)
        
        # Handle close long or short signal 
        elif row['strategy_signal'] in [StrategySignal.CLOSE_LONG, StrategySignal.CLOSE_SHORT] and position is not None:
            close_position(data, index, row, curr_qty, curr_balance, position)
        
        else:
            data.loc[index, 'qty'] = curr_qty
            data.loc[index, 'balance'] = curr_balance
        
    
    # Calculate portfolio value
    data['portfolio_value'] = data['Close'] * data['qty'] + data['balance']
    return data



