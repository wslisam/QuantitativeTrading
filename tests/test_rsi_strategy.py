import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
from strategies.rsi_strategy import rsi_strategy
from config import RSI_WINDOW, RSI_OVERBOUGHT, RSI_OVERSOLD

def test_rsi_strategy():
    # Create sample data that ensures both overbought and oversold conditions
    np.random.seed(42)  # Set seed for reproducibility
    dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
    prices = [100]
    for _ in range(199):
        change = np.random.normal(0, 3)
        if len(prices) % 40 == 0:  # Force some large moves to trigger signals
            change = 10 if len(prices) % 80 == 0 else -10
        prices.append(max(prices[-1] + change, 1))  # Ensure price doesn't go negative
    
    data = pd.DataFrame({
        'Close': prices
    }, index=dates)

    # Run the strategy
    signals = rsi_strategy(data)

    # Check if the signals are generated correctly
    assert len(signals) == len(data)
    assert 'signal' in signals.columns
    assert 'rsi' in signals.columns

    # Check if there are both buy and sell signals
    assert (signals['signal'] == 1).any(), "No buy signals generated"
    assert (signals['signal'] == -1).any(), "No sell signals generated"

    # Check if RSI is calculated correctly
    assert signals['rsi'].max() <= 100
    assert signals['rsi'].min() >= 0

    # Check if RSI values trigger correct signals
    assert (signals.loc[signals['rsi'] > RSI_OVERBOUGHT, 'signal'] == -1).any(), "No sell signals on overbought condition"
    assert (signals.loc[signals['rsi'] < RSI_OVERSOLD, 'signal'] == 1).any(), "No buy signals on oversold condition"

if __name__ == "__main__":
    pytest.main()