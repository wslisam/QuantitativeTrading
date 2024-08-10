import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from numpy.testing import assert_almost_equal

import pytest
import pandas as pd
import numpy as np
from strategies.macd_strategy import macd_strategy, execute_macd_strategy
from config import MACD_FAST, MACD_SLOW, MACD_SIGNAL, INITIAL_CAPITAL

def test_macd_strategy():
    # Create sample data
    dates = pd.date_range(start='2020-01-01', periods=300, freq='D')
    data = pd.DataFrame({
        'Close': np.random.randn(300).cumsum() + 100
    }, index=dates)

    # Run the strategy
    signals = macd_strategy(data)

    # Check if the signals are generated correctly
    assert len(signals) == len(data)
    assert 'signal' in signals.columns
    assert 'macd' in signals.columns  
    assert 'signal_line' in signals.columns
    assert 'price' in signals.columns  
    assert 'positions' in signals.columns  

    # Check if there are both buy and sell signals
    assert (signals['signal'] == 1).any()
    assert (signals['signal'] == -1).any()

    # Check if MACD components are calculated correctly
    ema_fast = data['Close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    assert_almost_equal(signals['macd'].iloc[-1], macd_line.iloc[-1], decimal=5)

    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    assert_almost_equal(signals['signal_line'].iloc[-1], signal_line.iloc[-1], decimal=5)

    # Check if signals are generated when MACD crosses the signal line
    for i in range(1, len(signals)):
        if signals['macd'].iloc[i-1] <= signals['signal_line'].iloc[i-1] and \
           signals['macd'].iloc[i] > signals['signal_line'].iloc[i]:
            assert signals['signal'].iloc[i] == 1
        elif signals['macd'].iloc[i-1] >= signals['signal_line'].iloc[i-1] and \
             signals['macd'].iloc[i] < signals['signal_line'].iloc[i]:
            assert signals['signal'].iloc[i] == -1

def test_execute_macd_strategy():
    # Create sample data
    dates = pd.date_range(start='2020-01-01', periods=300, freq='D')
    data = pd.DataFrame({
        'Close': np.random.randn(300).cumsum() + 100
    }, index=dates)

    # Run the execute strategy
    signals = execute_macd_strategy(data)

    # Check if the signals DataFrame is returned correctly
    assert len(signals) == len(data)
    assert 'cumulative_strategy_returns' in signals.columns
    assert 'strategy_returns' in signals.columns

    # Check if cumulative returns are calculated correctly
    assert signals['cumulative_strategy_returns'].iloc[0] == INITIAL_CAPITAL
    assert (signals['cumulative_strategy_returns'].iloc[-1] >= 0)

def test_empty_data():
    # Test with empty DataFrame
    data = pd.DataFrame(columns=['Close'])
    signals = macd_strategy(data)
    assert signals.empty

def test_insufficient_data():
    # Test with insufficient data
    data = pd.DataFrame({'Close': [100, 101]}, index=pd.date_range(start='2020-01-01', periods=2))
    signals = macd_strategy(data)
    assert 'signal' in signals.columns
    assert signals['signal'].isnull().all()  # Check if all values in 'signal' are NaN
    assert signals['macd'].isnull().all()    # Check if all values in 'macd' are NaN
    assert signals['signal_line'].isnull().all()  # Check if all values in 'signal_line' are NaN

def test_custom_macd_parameters():
    # Test with custom MACD parameters
    custom_fast = 5
    custom_slow = 20
    custom_signal = 9
    dates = pd.date_range(start='2020-01-01', periods=300, freq='D')
    data = pd.DataFrame({
        'Close': np.random.randn(300).cumsum() + 100
    }, index=dates)

    signals = macd_strategy(data, fast=custom_fast, slow=custom_slow, signal=custom_signal)
    assert 'macd' in signals.columns
    assert 'signal_line' in signals.columns

    # Check if the MACD line is calculated
    ema_fast = data['Close'].ewm(span=custom_fast, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=custom_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    assert_almost_equal(signals['macd'].iloc[-1], macd_line.iloc[-1], decimal=5)