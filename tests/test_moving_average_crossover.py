import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from numpy.testing import assert_almost_equal

import pytest
import pandas as pd
import numpy as np
from strategies.moving_average_crossover import moving_average_crossover
from config import MOVING_AVERAGE_SHORT_WINDOW, MOVING_AVERAGE_LONG_WINDOW

def test_moving_average_crossover():
    # Create sample data
    dates = pd.date_range(start='2020-01-01', periods=300, freq='D')
    data = pd.DataFrame({
        'Close': np.random.randn(300).cumsum() + 100
    }, index=dates)

    # Run the strategy
    signals = moving_average_crossover(data)

    # Check if the signals are generated correctly
    assert len(signals) == len(data)
    assert 'signal' in signals.columns
    assert 'short_mavg' in signals.columns
    assert 'long_mavg' in signals.columns

    # Check if there are both buy and sell signals
    assert (signals['signal'] == 1).any()
    assert (signals['signal'] == -1).any()

    # Check if moving averages are calculated correctly
    short_window_mean = data['Close'].iloc[:MOVING_AVERAGE_SHORT_WINDOW].mean()
    assert_almost_equal(signals['short_mavg'].iloc[MOVING_AVERAGE_SHORT_WINDOW-1], short_window_mean, decimal=5)
    
    long_window_mean = data['Close'].iloc[:MOVING_AVERAGE_LONG_WINDOW].mean()
    assert_almost_equal(signals['long_mavg'].iloc[MOVING_AVERAGE_LONG_WINDOW-1], long_window_mean, decimal=5)

if __name__ == "__main__":
    pytest.main()