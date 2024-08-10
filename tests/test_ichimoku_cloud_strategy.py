# tests/test_ichimoku_cloud_strategy.py

import pandas as pd
import numpy as np
import pytest
from strategies.ichimoku_cloud_strategy import calculate_ichimoku_cloud, ichimoku_cloud_strategy

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)  # Set seed for reproducibility
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
    prices = np.random.randn(len(dates)).cumsum() + 100
    return pd.DataFrame({
        'close': prices,
        'high': prices + np.random.rand(len(dates)),
        'low': prices - np.random.rand(len(dates))
    }, index=dates)

def test_calculate_ichimoku_cloud(sample_data):
    ichimoku = calculate_ichimoku_cloud(sample_data)
    
    assert isinstance(ichimoku, pd.DataFrame)
    assert 'conversion_line' in ichimoku.columns
    assert 'base_line' in ichimoku.columns
    assert 'leading_span_a' in ichimoku.columns
    assert 'leading_span_b' in ichimoku.columns
    assert 'lagging_span' in ichimoku.columns
    assert len(ichimoku) == len(sample_data)

def test_ichimoku_cloud_strategy(sample_data):
    signals = ichimoku_cloud_strategy(sample_data)
    
    assert isinstance(signals, pd.DataFrame)
    assert 'signal' in signals.columns
    assert 'positions' in signals.columns
    assert 'conversion_line' in signals.columns
    assert 'base_line' in signals.columns
    assert 'leading_span_a' in signals.columns
    assert 'leading_span_b' in signals.columns
    
    # Check if signals are either -1, 0, or 1
    assert set(signals['signal'].unique()).issubset({-1, 0, 1})
    
    # Check if positions are -2, -1, 0, 1, or 2
    assert set(signals['positions'].unique()).issubset({-2, -1, 0, 1, 2})