import pandas as pd
import numpy as np
from config import RSI_WINDOW, RSI_OVERBOUGHT, RSI_OVERSOLD

def calculate_rsi(data, window=RSI_WINDOW):
    """
    Calculate the Relative Strength Index (RSI).
    """
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def rsi_strategy(data, window=RSI_WINDOW, overbought=RSI_OVERBOUGHT, oversold=RSI_OVERSOLD):
    """
    Generate buy and sell signals using the RSI strategy.
    """
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    
    # Calculate RSI
    signals['rsi'] = calculate_rsi(data, window)

    # Create signals
    signals['signal'] = 0.0
    signals.loc[signals['rsi'] < oversold, 'signal'] = 1.0  # Buy signal
    signals.loc[signals['rsi'] > overbought, 'signal'] = -1.0  # Sell signal

    # Generate trading orders
    signals['positions'] = signals['signal'].diff()

    return signals