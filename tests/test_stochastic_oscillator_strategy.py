# tests/test_stochastic_oscillator_strategy.py

import pytest
import pandas as pd
import numpy as np
from strategies.stochastic_oscillator_strategy import (
    calculate_stochastic_oscillator,
    stochastic_oscillator_strategy
)
from config import (
    STOCHASTIC_K_PERIOD,
    STOCHASTIC_D_PERIOD,
    STOCHASTIC_OVERBOUGHT,
    STOCHASTIC_OVERSOLD
)

@pytest.fixture
def sample_data():
    # Create sample data
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
    np.random.seed(42)  # Set seed for reproducibility
    prices = np.random.randn(len(dates)).cumsum() + 100
    return pd.DataFrame({
        'close': prices,
        'high': prices + np.random.rand(len(dates)),
        'low': prices - np.random.rand(len(dates))
    }, index=dates)

def test_calculate_stochastic_oscillator(sample_data):
    stoch = calculate_stochastic_oscillator(sample_data)
    
    assert isinstance(stoch, pd.DataFrame)
    assert '%K' in stoch.columns
    assert '%D' in stoch.columns
    assert len(stoch) == len(sample_data)
    assert (stoch >= 0).all().all() and (stoch <= 100).all().all()

def test_stochastic_oscillator_strategy(sample_data):
    signals = stochastic_oscillator_strategy(sample_data)

    assert isinstance(signals, pd.DataFrame)
    assert 'signal' in signals.columns
    assert 'positions' in signals.columns
    assert '%K' in signals.columns
    assert '%D' in signals.columns

    # Check if signals are either -1, 0, or 1
    unique_signals = set(signals['signal'].unique())
    print("Unique Signals:", unique_signals)
    assert unique_signals.issubset({-1, 0, 1})

    # Check if positions are -2, -1, 0, 1, or 2
    unique_positions = set(signals['positions'].unique())
    print("Unique Positions:", unique_positions)
    assert unique_positions.issubset({-2, -1, 0, 1, 2})

def test_empty_data():
    empty_df = pd.DataFrame(columns=['close', 'high', 'low'])
    with pytest.raises(ValueError, match='Input data is empty'):
        calculate_stochastic_oscillator(empty_df)

def test_insufficient_data():
    small_df = pd.DataFrame({
        'close': [100, 102],
        'high': [101, 103],
        'low': [99, 101]
    }, index=pd.date_range(start='2020-01-01', periods=2))
    
    with pytest.raises(ValueError, match='Not enough data to calculate'):
        calculate_stochastic_oscillator(small_df)

import pytest
import pandas as pd
from strategies.stochastic_oscillator_strategy import stochastic_oscillator_strategy

def test_stochastic_oscillator_strategy_with_empty_data():
    empty_df = pd.DataFrame(columns=['close', 'high', 'low'])
    with pytest.raises(ValueError, match="Input data is empty"):
        stochastic_oscillator_strategy(empty_df)

def test_stochastic_oscillator_strategy_with_insufficient_data():
    small_df = pd.DataFrame({
        'close': [100, 102],
        'high': [101, 103],
        'low': [99, 101]
    }, index=pd.date_range(start='2020-01-01', periods=2))
    
    with pytest.raises(ValueError, match="Not enough data to calculate"):
        stochastic_oscillator_strategy(small_df)