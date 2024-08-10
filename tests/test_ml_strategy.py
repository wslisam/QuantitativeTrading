# tests/test_ml_strategy.py

import pytest
import pandas as pd
import numpy as np
from strategies.ml_strategy import ml_strategy

@pytest.fixture
def sample_data():
    # Create sample data with 100 rows
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100)
    close_prices = np.random.randint(100, 200, size=100)
    return pd.DataFrame({
        'Close': close_prices,
        'Open': close_prices + np.random.randint(-5, 5, size=100),
        'High': close_prices + np.random.randint(0, 10, size=100),
        'Low': close_prices - np.random.randint(0, 10, size=100),
        'Volume': np.random.randint(1000, 2000, size=100)
    }, index=dates)

def test_ml_strategy(sample_data):
    results = ml_strategy(sample_data)
    
    assert 'Signal' in results.columns
    assert 'Position' in results.columns
    assert 'Strategy_Returns' in results.columns
    assert len(results) < len(sample_data)  # Results should be shorter due to train-test split

def test_insufficient_data():
    insufficient_data = pd.DataFrame({
        'Close': [150] * 10,
        'Open': [150] * 10,
        'High': [155] * 10,
        'Low': [145] * 10,
        'Volume': [1500] * 10
    })
    
    with pytest.raises(ValueError):
        ml_strategy(insufficient_data)  # Try with only 10 data points