# strategies/ichimoku_cloud_strategy.py

import pandas as pd
import numpy as np
from config import (CONVERSION_LINE_PERIOD, BASE_LINE_PERIOD,
                    LEADING_SPAN_B_PERIOD, LAGGING_SPAN_PERIOD)

def calculate_ichimoku_cloud(data: pd.DataFrame, 
                              conversion_line_period: int = CONVERSION_LINE_PERIOD,
                              base_line_period: int = BASE_LINE_PERIOD,
                              leading_span_b_period: int = LEADING_SPAN_B_PERIOD,
                              lagging_span_period: int = LAGGING_SPAN_PERIOD) -> pd.DataFrame:
    """
    Calculate the Ichimoku Cloud components.
    
    Args:
        data (pd.DataFrame): DataFrame with 'high' and 'low' price columns.
        conversion_line_period (int): Period for Tenkan-sen (Conversion Line).
        base_line_period (int): Period for Kijun-sen (Base Line).
        leading_span_b_period (int): Period for Senkou Span B.
        lagging_span_period (int): Period for Chikou Span (Lagging Span).
        
    Returns:
        pd.DataFrame: DataFrame with Ichimoku Cloud components.
    """
    ichimoku = pd.DataFrame(index=data.index)
    
    # Tenkan-sen (Conversion Line)
    period_high = data['high'].rolling(window=conversion_line_period).max()
    period_low = data['low'].rolling(window=conversion_line_period).min()
    ichimoku['conversion_line'] = (period_high + period_low) / 2
    
    # Kijun-sen (Base Line)
    period_high26 = data['high'].rolling(window=base_line_period).max()
    period_low26 = data['low'].rolling(window=base_line_period).min()
    ichimoku['base_line'] = (period_high26 + period_low26) / 2
    
    # Senkou Span A
    ichimoku['leading_span_a'] = ((ichimoku['conversion_line'] + ichimoku['base_line']) / 2).shift(base_line_period)
    
    # Senkou Span B
    period_high52 = data['high'].rolling(window=leading_span_b_period).max()
    period_low52 = data['low'].rolling(window=leading_span_b_period).min()
    ichimoku['leading_span_b'] = ((period_high52 + period_low52) / 2).shift(base_line_period)
    
    # Lagging Span
    ichimoku['lagging_span'] = data['close'].shift(-lagging_span_period)
    
    return ichimoku

def ichimoku_cloud_strategy(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate buy and sell signals using Ichimoku Cloud strategy.
    
    Args:
        data (pd.DataFrame): DataFrame with 'high', 'low', and 'close' price columns.
        
    Returns:
        pd.DataFrame: DataFrame with signals.
    """
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['close']
    
    ichimoku = calculate_ichimoku_cloud(data)
    signals = signals.join(ichimoku)
    
    # Create signals
    signals['signal'] = 0.0
    signals['signal'] = np.where((signals['price'] > signals['leading_span_a']) & 
                                 (signals['price'] > signals['leading_span_b']) & 
                                 (signals['conversion_line'] > signals['base_line']), 1.0, signals['signal'])  # Buy signal
    signals['signal'] = np.where((signals['price'] < signals['leading_span_a']) & 
                                 (signals['price'] < signals['leading_span_b']) & 
                                 (signals['conversion_line'] < signals['base_line']), -1.0, signals['signal'])  # Sell signal
    
    # Generate trading orders
    # After calculating signals
    signals['positions'] = signals['signal'].diff().fillna(0)

    # positions represent transitions:
    # If from 0 to 1 (buy), should be 1
    # If from 1 to -1 (sell), should be -2
    # If from -1 to 0 (exit sell), should be 1
    # If from 1 to 0 (exit buy), should be -1
    
    return signals
