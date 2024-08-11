import sys
import os
import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import yfinance as yf
import pandas as pd
import numpy as np
from strategies import *
# Function to load data
@st.cache_data
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Function to plot strategy results
def plot_strategy_results(data, signals, strategy_name):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=(f'{strategy_name} Strategy', 'Volume'), row_width=[0.7, 0.3])
    
    # Plot candlestick chart
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Price'), row=1, col=1)
    
    # Plot buy signals
    fig.add_trace(go.Scatter(x=signals[signals['positions'] == 1].index, y=signals[signals['positions'] == 1]['price'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'), name='Buy Signal'), row=1, col=1)
    
    # Plot sell signals
    fig.add_trace(go.Scatter(x=signals[signals['positions'] == -1].index, y=signals[signals['positions'] == -1]['price'], mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'), name='Sell Signal'), row=1, col=1)
    
    # Plot volume
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume'), row=2, col=1)
    
    fig.update_layout(height=800, title_text=f"{strategy_name} Strategy Results")
    fig.update_xaxes(rangeslider_visible=False)
    
    return fig

# Streamlit app
def main():
    st.title('Trading Strategy Visualization')
    
    # Sidebar for user input
    st.sidebar.header('User Input')
    ticker = st.sidebar.text_input('Enter Stock Ticker', 'AAPL')
    start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2020-01-01'))
    end_date = st.sidebar.date_input('End Date', pd.to_datetime('2023-01-01'))
    
    # Load data
    data = load_data(ticker, start_date, end_date)
    
    # Strategy selection
    strategy = st.sidebar.selectbox('Select Strategy', 
                                    ['Moving Average Crossover', 'RSI', 'Bollinger Bands'])
    
    # Execute selected strategy
    if strategy == 'Moving Average Crossover':
        signals = execute_moving_average_crossover_strategy(data)
    elif strategy == 'RSI':
        signals = execute_rsi_strategy(data)
    elif strategy == 'Bollinger Bands':
        signals = execute_bollinger_bands_strategy(data)
    
    # Plot results
    fig = plot_strategy_results(data, signals, strategy)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display strategy performance
    if 'cumulative_strategy_returns' in signals.columns:
        st.subheader('Strategy Performance')
        st.write(f"Final Portfolio Value: ${signals['cumulative_strategy_returns'].iloc[-1]:.2f}")
        st.write(f"Total Return: {(signals['cumulative_strategy_returns'].iloc[-1] / 100000 - 1) * 100:.2f}%")

if __name__ == '__main__':
    main()