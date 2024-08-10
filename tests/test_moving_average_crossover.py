import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from strategies.moving_average_crossover import moving_average_crossover, execute_moving_average_crossover
from config import MOVING_AVERAGE_SHORT_WINDOW, MOVING_AVERAGE_LONG_WINDOW, INITIAL_CAPITAL

def test_moving_average_crossover():
    # Create sample data
    np.random.seed(42)  # Set seed for reproducibility
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
    assert_almost_equal(signals['short_mavg'].iloc[MOVING_AVERAGE_SHORT_WINDOW - 1], short_window_mean, decimal=5)
    
    long_window_mean = data['Close'].iloc[:MOVING_AVERAGE_LONG_WINDOW].mean()
    assert_almost_equal(signals['long_mavg'].iloc[MOVING_AVERAGE_LONG_WINDOW - 1], long_window_mean, decimal=5)

@pytest.fixture
def sample_data():
    # Create sample data for the execute test
    dates = pd.date_range(start='2020-01-01', periods=300, freq='D')
    return pd.DataFrame({
        'Close': np.random.randn(300).cumsum() + 100
    }, index=dates)

def test_execute_ma_crossover_strategy(sample_data):
    signals = execute_moving_average_crossover(sample_data)

    assert isinstance(signals, pd.DataFrame)
    assert 'price' in signals.columns
    assert 'short_mavg' in signals.columns
    assert 'long_mavg' in signals.columns
    assert 'signal' in signals.columns
    assert 'positions' in signals.columns
    assert 'cumulative_strategy_returns' in signals.columns

    # Check for NaN values in cumulative_strategy_returns
    assert signals['cumulative_strategy_returns'].isnull().sum() == 0, "Cumulative strategy returns contain NaN values."

    # Check if cumulative strategy returns are calculated correctly
    assert (signals['cumulative_strategy_returns'] >= 0).all()  # Ensure non-negative
    
    # Check if final cumulative strategy return is different from initial capital
    assert signals['cumulative_strategy_returns'].iloc[-1] != INITIAL_CAPITAL