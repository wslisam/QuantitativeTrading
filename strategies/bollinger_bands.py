import pandas as pd
import numpy as np
from config import BOLLINGER_WINDOW, BOLLINGER_NUM_STD, INITIAL_CAPITAL

def calculate_bollinger_bands(data: pd.DataFrame, window: int = BOLLINGER_WINDOW,
                              num_std: float = BOLLINGER_NUM_STD) -> tuple:
    """Calculate the Bollinger Bands."""
    middle = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    
    return middle, upper, lower

def bollinger_bands(data: pd.DataFrame, window: int = BOLLINGER_WINDOW,
                    num_std: float = BOLLINGER_NUM_STD) -> pd.DataFrame:
    """Generate buy and sell signals using Bollinger Bands strategy."""
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close'].ffill() # Forward fill to handle NaNs
    signals['middle'], signals['upper'], signals['lower'] = calculate_bollinger_bands(data, window, num_std)

    # Initialize the signal column
    signals['signal'] = 0.0
    signals.loc[signals['price'] < signals['lower'], 'signal'] = 1.0  # Buy signal
    signals.loc[signals['price'] > signals['upper'], 'signal'] = -1.0  # Sell signal

    # Generate trading orders
    signals['positions'] = signals['signal'].diff().fillna(0)

    return signals

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def execute_bollinger_bands_strategy(data: pd.DataFrame, initial_capital: float = INITIAL_CAPITAL) -> pd.DataFrame:
    """Execute the Bollinger Bands trading strategy."""
    signals = bollinger_bands(data)

    if signals.empty:
        raise ValueError("The signals DataFrame is empty.")

    shares = 0
    capital = initial_capital
    cumulative_returns = [initial_capital]

    for index, row in signals.iterrows():
        price = row['price']

        if pd.isna(price):  # Skip NaN prices
            cumulative_returns.append(cumulative_returns[-1])
            continue

        total_value = capital + (shares * price)
        cumulative_returns.append(total_value)

        if row['positions'] == 1:  # Buy signal
            shares_to_buy = capital // price
            if shares_to_buy > 0:
                shares += shares_to_buy
                capital -= shares_to_buy * price

        elif row['positions'] == -1:  # Sell signal
            if shares > 0:
                capital += shares * price
                shares = 0

    # Log lengths for debugging
    logger.info(f"Length of signals: {len(signals)}")
    logger.info(f"Length of cumulative_returns: {len(cumulative_returns)}")

    if len(signals) != len(cumulative_returns):
        logger.warning("Mismatch in lengths of signals and cumulative_returns")

    # Ensure cumulative_returns matches signals index length
    signals['cumulative_returns'] = cumulative_returns[:len(signals)]

    # Calculate strategy returns
    signals['strategy_returns'] = (signals['cumulative_returns'] - initial_capital) / initial_capital
    signals['cumulative_strategy_returns'] = signals['cumulative_returns']

    # Final return percentage
    total_return = (signals['cumulative_returns'].iloc[-1] - initial_capital) / initial_capital * 100
    logger.info(f"Total Return: {total_return:.2f}%")

    return signals