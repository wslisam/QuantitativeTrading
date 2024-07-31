import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
from strategies.rsi_strategy import rsi_strategy

def test_rsi_strategy():
    # Create sample data
    data = pd.DataFrame({
        'Close': [10, 11, 12, 13, 14, 15, 14, 13, 12, 11] * 10
    })
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
from strategies.rsi_strategy import rsi_strategy
from config import RSI_WINDOW, RSI_OVERBOUGHT, RSI_OVERSOLD

def test_rsi_strategy():
    # Create sample data
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'Close': np.random.randn(100).cumsum() + 100
    }, index=dates)

    # Run the strategy
    signals = rsi_strategy(data)

    # Check if the signals are generated correctly
    assert len(signals) == len(data)
    assert 'signal' in signals.columns
    assert 'rsi' in signals.columns  # Changed from 'RSI' to 'rsi'

    # Check if there are both buy and sell signals
    assert (signals['signal'] == 1).any()
    assert (signals['signal'] == -1).any()

    # Check if RSI is calculated correctly
    assert signals['rsi'].max() <= 100
    assert signals['rsi'].min() >= 0

if __name__ == "__main__":
    pytest.main()
    # Run the strategy
    signals = rsi_strategy(data)

    # Check if the signals are generated correctly
    assert len(signals) == len(data)
    assert 'signal' in signals.columns
    assert 'positions' in signals.columns
    assert 'RSI' in signals.columns

    # Check if RSI values are within the expected range
    assert signals['RSI'].min() >= 0
    assert signals['RSI'].max() <= 100

    # Add more specific assertions based on expected behavior

if __name__ == "__main__":
    pytest.main()