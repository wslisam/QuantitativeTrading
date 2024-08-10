# strategies/stochastic_oscillator_strategy.py

import pandas as pd
import numpy as np
from config import STOCHASTIC_K_PERIOD, STOCHASTIC_D_PERIOD, STOCHASTIC_OVERBOUGHT, STOCHASTIC_OVERSOLD

def calculate_stochastic_oscillator(data, k_period=STOCHASTIC_K_PERIOD, d_period=STOCHASTIC_D_PERIOD):
    """
    Calculate the Stochastic Oscillator
    
    :param data: DataFrame with 'high', 'low', and 'close' price columns
    :param k_period: The period for %K line
    :param d_period: The period for %D line (signal line)
    :return: DataFrame with %K and %D values
    """
    if data.empty:
        raise ValueError("Input data is empty")

    if len(data) < max(k_period, d_period):
        raise ValueError("Not enough data to calculate")

    low_min = data['low'].rolling(window=k_period, min_periods=1).min()
    high_max = data['high'].rolling(window=k_period, min_periods=1).max()

    # Calculate %K
    k_line = 100 * (data['close'] - low_min) / (high_max - low_min)
    k_line = k_line.clip(0, 100)  # Ensure values are between 0 and 100

    # Calculate %D
    d_line = k_line.rolling(window=d_period, min_periods=1).mean()

    return pd.DataFrame({'%K': k_line, '%D': d_line})

def stochastic_oscillator_strategy(data, k_period=STOCHASTIC_K_PERIOD, d_period=STOCHASTIC_D_PERIOD, overbought=STOCHASTIC_OVERBOUGHT, oversold=STOCHASTIC_OVERSOLD):
    """
    Generate buy and sell signals using Stochastic Oscillator strategy.
    
    :param data: DataFrame with 'high', 'low', and 'close' price columns
    :param k_period: The period for %K line
    :param d_period: The period for %D line (signal line)
    :param overbought: The overbought threshold
    :param oversold: The oversold threshold
    :return: DataFrame with signals
    """
    if data.empty:
        raise ValueError("Input data is empty")

    if len(data) < max(k_period, d_period):
        raise ValueError("Not enough data to calculate")

    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['close']
    
    stoch = calculate_stochastic_oscillator(data, k_period, d_period)
    signals['%K'] = stoch['%K']
    signals['%D'] = stoch['%D']
    
    # Create signals
    signals['signal'] = 0.0
    signals['signal'] = np.where((signals['%K'] < oversold) & (signals['%D'] < oversold), 1.0, signals['signal'])  # Buy signal
    signals['signal'] = np.where((signals['%K'] > overbought) & (signals['%D'] > overbought), -1.0, signals['signal'])  # Sell signal
    
    # Generate trading orders
    signals['positions'] = signals['signal'].diff().fillna(0)  # Fill NaN with 0 for the first row
    
    return signals