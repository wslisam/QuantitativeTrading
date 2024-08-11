import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from strategies.bollinger_bands import bollinger_bands, execute_bollinger_bands_strategy, calculate_bollinger_bands
from config import BOLLINGER_WINDOW, BOLLINGER_NUM_STD, INITIAL_CAPITAL

def create_sample_data():
    """Generate sample price data for testing."""
    np.random.seed(42)  # Set seed for reproducibility
    dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
    prices = [100]
    for _ in range(199):
        change = np.random.normal(0, 2)
        prices.append(max(prices[-1] + change, 1))  # Ensure price doesn't go negative

    return pd.DataFrame({'Close': prices}, index=dates)

def test_bollinger_bands():
    # Create sample data
    data = create_sample_data()

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

def test_execute_bollinger_bands_strategy():
    # Create sample data
    data = create_sample_data()

    # Run the strategy
    signals = execute_bollinger_bands_strategy(data)  # Changed this line

    # Check if cumulative returns and strategy returns are calculated
    assert 'cumulative_returns' in signals.columns
    assert 'strategy_returns' in signals.columns
    assert 'cumulative_strategy_returns' in signals.columns

    # Check lengths
    assert len(signals['cumulative_returns']) == len(signals)

    # Check if initial capital is correctly managed
    initial_capital = INITIAL_CAPITAL
    assert signals['cumulative_returns'].iloc[0] == initial_capital, "Initial capital not set correctly."
    assert signals['strategy_returns'].iloc[0] == 0, "Initial strategy return should be zero."

    # Check for NaN values
    assert signals['cumulative_returns'].isnull().sum() == 0, "Cumulative returns contain NaN values."
    assert signals['strategy_returns'].isnull().sum() == 0, "Strategy returns contain NaN values."
    assert signals['cumulative_strategy_returns'].isnull().sum() == 0, "Cumulative strategy returns contain NaN values."

    # Ensure cumulative returns change
    assert signals['cumulative_returns'].iloc[-1] != initial_capital, "Cumulative returns did not change."
    
    # Check for negative returns
    if signals['cumulative_returns'].iloc[-1] < initial_capital:
        assert signals['strategy_returns'].iloc[-1] < 0, "Strategy return should reflect negative performance."

    # Check for signals consistency
    assert (signals['positions'].isnull().sum() == 0), "Positions contain NaN values."
    assert signals['positions'].nunique() > 1, "Positions should reflect valid trades."

def test_calculate_bollinger_bands():
    data = create_sample_data()  # Create sample data
    middle, upper, lower = calculate_bollinger_bands(data)

    # Check if lengths match
    assert len(middle) == len(data)
    assert len(upper) == len(data)
    assert len(lower) == len(data)

def test_bollinger_bands_with_small_data():
    small_data = pd.DataFrame({'Close': [100, 101]})
    signals = bollinger_bands(small_data)

    # Ensure signals are generated correctly
    assert signals['signal'].nunique() <= 2  # Only buy/sell/no signal

def test_no_signals():
    constant_data = pd.DataFrame({'Close': [100] * 20})
    signals = bollinger_bands(constant_data)

    # Ensure no buy/sell signals are generated
    assert (signals['signal'] == 0).all(), "Should have no buy/sell signals."

def test_no_trades():
    # Create constant price data where no trades should occur
    constant_data = pd.DataFrame({'Close': [100] * 20})
    
    # Run the strategy
    signals = execute_bollinger_bands_strategy(constant_data)

    # Check if cumulative_returns contains the initial value
    assert (signals['cumulative_returns'] == INITIAL_CAPITAL).all(), "Cumulative returns should remain constant when no trades occur."
    assert (signals['strategy_returns'] == 0).all(), "Strategy returns should be zero when no trades occur."
    assert len(signals['cumulative_returns']) == len(signals), "Cumulative returns length should match signals length."
    