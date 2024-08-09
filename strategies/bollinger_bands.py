import pandas as pd
import numpy as np
from config import BOLLINGER_WINDOW, BOLLINGER_NUM_STD

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