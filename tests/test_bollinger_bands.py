import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
from strategies.bollinger_bands import bollinger_bands
from config import BOLLINGER_WINDOW, BOLLINGER_NUM_STD

def test_bollinger_bands():
    # Create sample data that ensures both upper and lower band crossings
    np.random.seed(42)  # Set seed for reproducibility
    dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
    prices = [100]
    for _ in range(199):
        change = np.random.normal(0, 2)
        if len(prices) % 40 == 0:  # Force some large moves to trigger signals
            change = 8 if len(prices) % 80 == 0 else -8
        prices.append(max(prices[-1] + change, 1))  # Ensure price doesn't go negative
    
    data = pd.DataFrame({
        'Close': prices
    }, index=dates)

    # Run the strategy
    signals = bollinger_bands(data)

    # Check if the signals are generated correctly
    assert len(signals) == len(data)
    assert 'signal' in signals.columns
    assert 'upper' in signals.columns
    assert 'lower' in signals.columns
    assert 'middle' in signals.columns

    # Check if there are both buy and sell signals
    assert (signals['signal'] == 1).any(), "No buy signals generated"
    assert (signals['signal'] == -1).any(), "No sell signals generated"

    # Check if Bollinger Bands are calculated correctly
    assert np.isclose(signals['middle'].iloc[BOLLINGER_WINDOW-1], data['Close'].iloc[:BOLLINGER_WINDOW].mean(), rtol=1e-5)
    
    expected_upper = (
        signals['middle'].iloc[BOLLINGER_WINDOW-1] + 
        BOLLINGER_NUM_STD * data['Close'].iloc[:BOLLINGER_WINDOW].std()
    )
    assert np.isclose(signals['upper'].iloc[BOLLINGER_WINDOW-1], expected_upper, rtol=1e-5)
    
    expected_lower = (
        signals['middle'].iloc[BOLLINGER_WINDOW-1] - 
        BOLLINGER_NUM_STD * data['Close'].iloc[:BOLLINGER_WINDOW].std()
    )
    assert np.isclose(signals['lower'].iloc[BOLLINGER_WINDOW-1], expected_lower, rtol=1e-5)

if __name__ == "__main__":
    pytest.main()