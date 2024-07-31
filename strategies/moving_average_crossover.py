import pandas as pd
import numpy as np
from config import MOVING_AVERAGE_SHORT_WINDOW, MOVING_AVERAGE_LONG_WINDOW

def moving_average_crossover(data, short_window=MOVING_AVERAGE_SHORT_WINDOW, long_window=MOVING_AVERAGE_LONG_WINDOW):
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    signals['short_mavg'] = data['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
    signals['long_mavg'] = data['Close'].rolling(window=long_window, min_periods=1, center=False).mean()

    # Create signals
    signals['signal'] = 0.0
    signals.loc[signals['short_mavg'] > signals['long_mavg'], 'signal'] = 1.0

    # Generate trading orders
    signals['positions'] = signals['signal'].diff()

    return signals