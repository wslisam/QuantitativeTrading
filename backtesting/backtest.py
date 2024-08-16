# backtesting/backtest.py

import pandas as pd
import numpy as np
from config import *


class Backtest:
    def __init__(self, data, strategy):
        self.data = data
        self.strategy = strategy
        self.signals = None

    def run(self):
        self.signals = self.strategy(self.data)
        return self.signals

    def calculate_metrics(self, signals):
        initial_investment = INITIAL_CAPITAL
        final_value = signals['cumulative_strategy_returns'].iloc[-1]
        total_return = (final_value / initial_investment - 1) * 100

        num_trades = len(signals[signals['positions'] != 0])
        
        # Calculate Sharpe Ratio
        returns = signals['cumulative_strategy_returns'].pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        
        # Calculate Maximum Drawdown
        cumulative_returns = signals['cumulative_strategy_returns']
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        return {
            "initial_investment": initial_investment,
            "final_value": final_value,
            "total_return": total_return,
            "num_trades": num_trades,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown
        }