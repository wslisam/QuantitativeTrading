import pandas as pd
import numpy as np
from config import BOLLINGER_WINDOW, BOLLINGER_NUM_STD

def bollinger_bands(data, window=BOLLINGER_WINDOW, num_std=BOLLINGER_NUM_STD):
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    signals['middle'] = data['Close'].rolling(window=window).mean()
    signals['std'] = data['Close'].rolling(window=window).std()
    signals['upper'] = signals['middle'] + (signals['std'] * num_std)
    signals['lower'] = signals['middle'] - (signals['std'] * num_std)

    # Create signals
    signals['signal'] = 0.0
    signals.loc[signals['price'] < signals['lower'], 'signal'] = 1.0
    signals.loc[signals['price'] > signals['upper'], 'signal'] = -1.0

    # Generate trading orders
    signals['positions'] = signals['signal'].diff()

    return signals