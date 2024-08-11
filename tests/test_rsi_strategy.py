import sys
import os
import pytest
import pandas as pd
import numpy as np
from strategies.rsi_strategy import rsi_strategy, execute_rsi_strategy
from config import RSI_WINDOW, RSI_OVERBOUGHT, RSI_OVERSOLD, INITIAL_CAPITAL

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_sample_data():
    """Generate sample price data for testing."""
    np.random.seed(42)  # Set seed for reproducibility
    dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
    prices = [100]
    for _ in range(199):
        change = np.random.normal(0, 3)
        if len(prices) % 40 == 0:  # Force some large moves to trigger signals
            change = 10 if len(prices) % 80 == 0 else -10
        prices.append(max(prices[-1] + change, 1))  # Ensure price doesn't go negative

    return pd.DataFrame({'Close': prices}, index=dates)

@pytest.fixture
def sample_data():
    """Fixture to provide sample data for tests."""
    return create_sample_data()

def test_rsi_strategy(sample_data):
    # Run the strategy
    signals = rsi_strategy(sample_data)

    # Check if the signals are generated correctly
    assert len(signals) == len(sample_data)
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

    # Check for correct signal transitions
    if 'positions' in signals.columns:
        assert (signals['positions'].notna()).any(), "No trading positions generated"
        assert (signals['positions'].isin([1, -1])).any(), "Invalid trading positions detected"

def test_execute_rsi_strategy(sample_data):
    # Execute the strategy
    signals = execute_rsi_strategy(sample_data)

    # Check if the signals dataframe has the expected columns
    expected_columns = ['returns', 'strategy_returns', 'cumulative_returns', 'cumulative_strategy_returns']
    for column in expected_columns:
        assert column in signals.columns, f"{column} not found in signals DataFrame"

    # Ensure the DataFrame is not empty
    assert not signals.empty, "Signals DataFrame should not be empty"

    # Check initial capital is reflected correctly
    assert signals['cumulative_strategy_returns'].iloc[0] == INITIAL_CAPITAL, "Initial capital not set correctly"

    # Check cumulative returns are calculated correctly
    assert signals['cumulative_returns'].iloc[0] == INITIAL_CAPITAL, "Initial cumulative returns not set correctly"

    # Additional checks for returns consistency
    assert signals['returns'].notna().all(), "Returns column contains NaN values"
    assert signals['strategy_returns'].notna().all(), "Strategy returns column contains NaN values"