import yfinance as yf
import pandas as pd
import numpy as np

def fetch_data(symbol, start_date, end_date):
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data available for {symbol}")
        return data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        print(f"Attempted to fetch data from {start_date} to {end_date}")
        print("Using sample data for testing purposes.")
        return generate_sample_data(start_date, end_date)

def generate_sample_data(start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    close_prices = np.random.randn(len(date_range)).cumsum() + 100
    return pd.DataFrame({'Close': close_prices}, index=date_range)