import pandas as pd
import numpy as np
from config import MACD_FAST, MACD_SLOW, MACD_SIGNAL

def macd_strategy(data, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    
    # Calculate MACD
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    
    signals['macd'] = macd
    signals['signal_line'] = signal_line
    
    # Create signals
    signals['signal'] = 0.0
    signals.loc[signals['macd'] > signals['signal_line'], 'signal'] = 1.0
    
    # Generate trading orders
    signals['positions'] = signals['signal'].diff()
    
    return signals