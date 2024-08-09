import pandas as pd
import numpy as np
from config import MACD_FAST, MACD_SLOW, MACD_SIGNAL

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
    signals['macd'], signals['signal_line'] = calculate_macd(data, fast, slow, signal)

    # Create signals
    signals['signal'] = 0.0
    signals.loc[signals['macd'] > signals['signal_line'], 'signal'] = 1.0  # Buy signal
    signals.loc[signals['macd'] <= signals['signal_line'], 'signal'] = -1.0  # Sell signal

    # Generate trading orders
    signals['positions'] = signals['signal'].diff()

    return signals