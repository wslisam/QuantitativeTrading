import pandas as pd
import numpy as np
from config import BOLLINGER_WINDOW, BOLLINGER_NUM_STD, INITIAL_CAPITAL

def calculate_bollinger_bands(data, window=BOLLINGER_WINDOW, num_std=BOLLINGER_NUM_STD):
    """
    Calculate Bollinger Bands.
    """
    middle = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    
    return middle, upper, lower

def bollinger_bands(data, window=BOLLINGER_WINDOW, num_std=BOLLINGER_NUM_STD):
    """
    Generate buy and sell signals using the Bollinger Bands strategy.
    """
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    
    # Calculate Bollinger Bands
    signals['middle'], signals['upper'], signals['lower'] = calculate_bollinger_bands(data, window, num_std)

    # Create signals
    signals['signal'] = 0.0
    signals.loc[signals['price'] < signals['lower'], 'signal'] = 1.0  # Buy signal
    signals.loc[signals['price'] > signals['upper'], 'signal'] = -1.0  # Sell signal

    # Generate trading orders
    signals['positions'] = signals['signal'].diff()

    return signals

def execute_bollinger_bands_strategy(signals):
    """
    Execute the trading strategy based on Bollinger Bands signals.
    """
    initial_capital = INITIAL_CAPITAL  # Starting capital
    shares = 0  # Number of shares held
    cumulative_returns = []
    cumulative_strategy_returns = []

    for index, row in signals.iterrows():
        if row['positions'] == 1:  # Buy signal
            shares += initial_capital // row['price']  # Buy as many shares as possible
            initial_capital -= shares * row['price']
        elif row['positions'] == -1:  # Sell signal
            initial_capital += shares * row['price']  # Sell all shares
            shares = 0
        
        # Calculate total portfolio value
        total_value = initial_capital + (shares * row['price'])
        cumulative_returns.append(total_value)

        # Calculate cumulative strategy returns (percentage change)
        if total_value > INITIAL_CAPITAL:
            cumulative_strategy_returns.append((total_value - INITIAL_CAPITAL) / INITIAL_CAPITAL)
        else:
            cumulative_strategy_returns.append(0)

    signals['cumulative_returns'] = cumulative_returns
    signals['cumulative_strategy_returns'] = cumulative_strategy_returns
    return signals