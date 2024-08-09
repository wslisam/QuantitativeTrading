import pandas as pd
import numpy as np
from config import MOVING_AVERAGE_SHORT_WINDOW, MOVING_AVERAGE_LONG_WINDOW

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