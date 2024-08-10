import pandas as pd
import numpy as np
from config import MACD_FAST, MACD_SLOW, MACD_SIGNAL, INITIAL_CAPITAL

def calculate_macd(data, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    """
    Calculate MACD and signal line.
    """
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    
    return macd, signal_line

def macd_strategy(data, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    """
    Generate buy and sell signals using the MACD strategy.
    """
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    
    # Calculate MACD and signal line
    # Ensure there is enough data to compute MACD
    if len(data) < max(fast, slow, signal):
        signals['signal'] = np.nan
        signals['macd'] = np.nan
        signals['signal_line'] = np.nan
        signals['positions'] = np.nan
        return signals
    
    signals['macd'], signals['signal_line'] = calculate_macd(data, fast, slow, signal)

    # Create signals
    signals['signal'] = 0.0
    signals.loc[signals['macd'] > signals['signal_line'], 'signal'] = 1.0  # Buy signal
    signals.loc[signals['macd'] <= signals['signal_line'], 'signal'] = -1.0  # Sell signal

    # Generate trading orders
    signals['positions'] = signals['signal'].diff()

    return signals

def execute_macd_strategy(data, initial_capital=INITIAL_CAPITAL, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    """
    Execute the MACD strategy and calculate returns.
    
    :param data: DataFrame with 'close' price column
    :param initial_capital: Initial capital for the strategy
    :param fast: Fast MACD window
    :param slow: Slow MACD window
    :param signal: Signal line window
    :return: DataFrame with strategy performance
    """
    signals = macd_strategy(data, fast, slow, signal)

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