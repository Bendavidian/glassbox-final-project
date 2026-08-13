import pandas as pd
import numpy  as np
from models_backtest  import ActionType, Position, PositionType, StrategySignal
from getVixData       import get_val_historical_data
from abc              import ABC, abstractmethod
from typing           import Tuple



class BaseStrategy(ABC):
    def __init__(self, sl_rate: float=None, tp_rate: float=None) -> None:
        super().__init__()
        self.sl_rate = sl_rate
        self.tp_rate = tp_rate
    
    @abstractmethod
    def calc_signal(self, data: pd.DataFrame):
        pass

    def calc_qty(self, real_price: float, balance: float, action: ActionType, **kwargs) -> float:
        if action == ActionType.BUY:
            qty = balance / real_price
        
        elif action == ActionType.SELL:
            qty =  balance / real_price
        
        return qty    
    
    def check_sl_tp(self, row: pd.Series, position: Position) -> Tuple[float, float, ActionType]:
        sl_res = self.is_stop_loss(row, position)
        if sl_res is not None:
            return sl_res
        
        tp_res = self.is_take_profit(row, position)
        if tp_res is not None:
            return tp_res
    
    def is_stop_loss(self, row: pd.Series, position: Position) -> Tuple[float, float, ActionType]:
        """
        Checks if the price has hit the stop-loss level.
        
        Returns:
            Tuple[float, float, ActionType] or None: If stop-loss is triggered, returns a tuple containing quantity and stop-loss price and action type, otherwise returns None.
        """
        if self.sl_rate is not None:
            long_stop_loss_price = position.price * (1 - self.sl_rate)
            if position.type == PositionType.LONG and row['Low'] >= long_stop_loss_price:
                return position.qty, long_stop_loss_price, ActionType.SELL
            
            short_stop_loss_price = position.price * (1 + self.sl_rate)
            if position.type == PositionType.SHORT and row['High'] <= short_stop_loss_price:
                return position.qty, short_stop_loss_price, ActionType.BUY
    
    def is_take_profit(self, row: pd.Series, position: Position) -> Tuple[float, float, ActionType]:
        """
        Checks if the price has hit the take-profit level.

        Returns:
            Tuple[float, float, ActionType] or None: If take-profit is triggered, returns a tuple containing quantity and take-profit price and action type, otherwise returns None.
        """
        if self.tp_rate is not None:
            long_take_profit_price = position.price * (1 + self.tp_rate)
            if position.type == PositionType.LONG and row['High'] <= long_take_profit_price:
                return position.qty, long_take_profit_price, ActionType.SELL
            
            short_take_profit_price = position.price * (1 - self.tp_rate)
            if position.type == PositionType.SHORT and row['Low'] >= short_take_profit_price:
                return position.qty, short_take_profit_price, ActionType.BUY
            

class LinearForcastingClassifiction(BaseStrategy):
    def __init__(self,exp,settings,sl_rate: float = None, tp_rate: float = None) -> pd.Series:
        super().__init__(sl_rate, tp_rate)
        self.exp      = exp
        self.settings = settings

    def calc_buy_signal(self, data: pd.DataFrame,stc) -> pd.Series:
        pass



    def calc_sell_signal(self, data: pd.DataFrame,stc) -> pd.Series:
        pass


    
    def calc_signal(self, data: pd.DataFrame) -> pd.Series:
        data['strategy_signal'] = StrategySignal.DO_NOTHING
        print('>>>>>>>Start Training>>>>>>>>>>>>>>>>>>>>>>>>>>')
        self.exp.train(self.settings)
        print('>>>>>>>End Training>>>>>>>>>>>>>>>>>>>>>>>>>>')
        hypers = {'confidence':list(range(11,12)), 'lowerbound':[i/8 for i in range(1,7)],'upperbound':[i for i in range(1,11)]}
        print('>>>>>>>Start Validating>>>>>>>>>>>>>>>>>>>>>>>>>>')
        arg_conf,arg_low,arg_up = self.exp.validate(self.settings,hypers)
        #arg_conf,arg_low,arg_up = 11,0.25,9
        print('>>>>>>>End Validatin>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('>>>>>>>Start Training Validate>>>>>>>>>>>>>>>>>>>>>>>>>>')
        self.exp.train(self.settings,True)
        print('>>>>>>>End Training Validate>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('>>>>>>>Start Testing>>>>>>>>>>>>>>>>>>>>>>>>>>')
        self.exp.test(self.settings,data,arg_conf,arg_low,arg_up)
        print('>>>>>>>End Testing>>>>>>>>>>>>>>>>>>>>>>>>>>')
