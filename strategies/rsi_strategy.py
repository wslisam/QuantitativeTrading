import pandas as pd
import numpy as np
from config import RSI_WINDOW, RSI_OVERBOUGHT, RSI_OVERSOLD, INITIAL_CAPITAL

def calculate_rsi(data: pd.DataFrame, period: int = RSI_WINDOW) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI).
    
    :param data: DataFrame with 'Close' price column
    :param period: The period over which to calculate the RSI
    :return: Series with RSI values
    """
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # Avoid division by zero
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def rsi_strategy(data: pd.DataFrame, period: int = RSI_WINDOW, 
                 overbought: float = RSI_OVERBOUGHT, oversold: float = RSI_OVERSOLD) -> pd.DataFrame:
    """
    Generate buy and sell signals using RSI strategy.
    
    :param data: DataFrame with 'Close' price column
    :param period: The period over which to calculate the RSI
    :param overbought: The overbought threshold
    :param oversold: The oversold threshold
    :return: DataFrame with signals
    """
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    signals['rsi'] = calculate_rsi(data, period)
    
    # Create signals
    signals['signal'] = np.where(signals['rsi'] < oversold, 1.0, 0.0)  # Buy signal
    signals['signal'] = np.where(signals['rsi'] > overbought, -1.0, signals['signal'])  # Sell signal
    
    # Generate trading orders
    signals['positions'] = signals['signal'].diff()
    
    # Fill NaN values
    signals.fillna(0, inplace=True)  # Replace NaNs with 0 instead of dropping rows
    
    return signals

def execute_rsi_strategy(data: pd.DataFrame, initial_capital: float = INITIAL_CAPITAL, 
                         period: int = RSI_WINDOW, overbought: float = RSI_OVERBOUGHT, 
                         oversold: float = RSI_OVERSOLD) -> pd.DataFrame:
    signals = rsi_strategy(data, period, overbought, oversold)

    signals['returns'] = data['Close'].pct_change().fillna(0) 

    shares = 0
    capital = initial_capital
    cumulative_returns = []
    strategy_returns = []

    for index, row in signals.iterrows():
        total_value = capital + (shares * row['price']) if shares > 0 else capital
        cumulative_returns.append(total_value)

        if row['signal'] == 1.0 and capital >= row['price']:  # Buy if sufficient capital
            shares += capital // row['price']
            capital -= shares * row['price']
        
        elif row['signal'] == -1.0:
            capital += shares * row['price']
            shares = 0

        daily_return = (total_value - initial_capital) / initial_capital
        strategy_returns.append(daily_return)

    signals['cumulative_returns'] = cumulative_returns
    signals['strategy_returns'] = strategy_returns
    signals['cumulative_strategy_returns'] = [initial_capital] + [total_value for total_value in cumulative_returns[1:]]

    return signals