import pandas as pd
import numpy as np
from config import MOVING_AVERAGE_SHORT_WINDOW, MOVING_AVERAGE_LONG_WINDOW, INITIAL_CAPITAL

def calculate_moving_averages(data, short_window=MOVING_AVERAGE_SHORT_WINDOW, long_window=MOVING_AVERAGE_LONG_WINDOW):
    """
    Calculate short and long moving averages.
    """
    short_mavg = data['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
    long_mavg = data['Close'].rolling(window=long_window, min_periods=1, center=False).mean()
    return short_mavg, long_mavg

def moving_average_crossover(data, short_window=MOVING_AVERAGE_SHORT_WINDOW, long_window=MOVING_AVERAGE_LONG_WINDOW):
    """
    Generate buy and sell signals using the Moving Average Crossover strategy.
    """
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    
    # Calculate moving averages
    signals['short_mavg'], signals['long_mavg'] = calculate_moving_averages(data, short_window, long_window)

    # Create signals
    signals['signal'] = 0.0
    signals.loc[signals['short_mavg'] > signals['long_mavg'], 'signal'] = 1.0  # Buy signal
    signals.loc[signals['short_mavg'] <= signals['long_mavg'], 'signal'] = -1.0  # Sell signal

    # Generate trading orders
    signals['positions'] = signals['signal'].diff()

    return signals

def execute_moving_average_crossover(data, initial_capital=INITIAL_CAPITAL, short_window=MOVING_AVERAGE_SHORT_WINDOW, long_window=MOVING_AVERAGE_LONG_WINDOW):
    """
    Execute the Moving Average Crossover strategy and calculate returns.
    
    :param data: DataFrame with 'close' price column
    :param initial_capital: Initial capital for the strategy
    :param short_window: Short-term moving average window
    :param long_window: Long-term moving average window
    :return: DataFrame with strategy performance
    """
    signals = moving_average_crossover(data, short_window, long_window)

    # Ensure 'positions' is initialized correctly
    if signals['positions'].isnull().any():
        print("Positions contain NaN values:", signals['positions'].isnull().sum())
    
    # Calculate returns
    signals['returns'] = signals['price'].pct_change()

    # Initialize strategy returns
    signals['strategy_returns'] = signals['positions'].shift(1) * signals['returns']

    # Fill NaN values in strategy returns
    signals['strategy_returns'] = signals['strategy_returns'].fillna(0)

    # Initialize cumulative returns
    signals['cumulative_returns'] = (1 + signals['strategy_returns']).cumprod()

    # Check for NaN in cumulative returns
    if signals['cumulative_returns'].isnull().any():
        print("Cumulative returns contain NaN values:", signals['cumulative_returns'].isnull().sum())
    
    # Calculate cumulative strategy returns
    signals['cumulative_strategy_returns'] = initial_capital * signals['cumulative_returns']
    
    return signals