import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import yfinance as yf
import pandas as pd
import numpy as np
from strategies import *
from datetime import timedelta

# Function to load data
@st.cache_data
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Function to plot strategy results
def plot_strategy_results(data, signals, strategy_name):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
                        subplot_titles=(f'{strategy_name} Strategy', 'Volume', 'Strategy Returns'), 
                        row_heights=[0.5, 0.2, 0.3])
    
    # Plot candlestick chart
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Price'), row=1, col=1)
    
    # Plot buy signals
    fig.add_trace(go.Scatter(x=signals[signals['positions'] == 1].index, y=signals[signals['positions'] == 1]['price'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'), name='Buy Signal'), row=1, col=1)
    
    # Plot sell signals
    fig.add_trace(go.Scatter(x=signals[signals['positions'] == -1].index, y=signals[signals['positions'] == -1]['price'], mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'), name='Sell Signal'), row=1, col=1)
    
    # Plot volume
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume'), row=2, col=1)
    
    # Plot strategy returns
    fig.add_trace(go.Scatter(x=signals.index, y=signals['cumulative_strategy_returns'], name='Cumulative Strategy Returns', line=dict(color='orange')), row=3, col=1)
    
    fig.update_layout(height=800, title_text=f"{strategy_name} Strategy Results", margin=dict(l=40, r=40, t=40, b=40))
    fig.update_xaxes(rangeslider_visible=False)
    
    return fig

# Function to calculate additional metrics
def calculate_metrics(signals):
    initial_investment = 100000
    final_value = signals['cumulative_strategy_returns'].iloc[-1]
    total_return = (final_value / initial_investment - 1) * 100
    
    num_trades = len(signals[signals['positions'] != 0])
    # win_rate = (signals['positions'] == 1).sum() / num_trades * 100
    
    # Calculate Sharpe Ratio
    returns = signals['cumulative_strategy_returns'].pct_change().dropna()
    sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
    
    # Calculate Maximum Drawdown
    cumulative_returns = signals['cumulative_strategy_returns']
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    return {
        "initial_investment": initial_investment,
        "final_value": final_value,
        "total_return": total_return,
        "num_trades": num_trades,
        # "win_rate": win_rate,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown
    }

# Streamlit app
def main():
    st.set_page_config(layout="wide", page_title="Trading Strategy Visualization")
    st.title('Trading Strategy Visualization')
    
    # Sidebar for user input
    st.sidebar.header('Strategy Configuration')
    ticker = st.sidebar.text_input('Enter Stock Ticker', 'AAPL', help="Enter a valid stock ticker symbol (e.g., AAPL, TSLA).")
    
    start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2020-01-01'))
    end_date = st.sidebar.date_input('End Date', pd.to_datetime('2023-01-01'))

    if end_date < start_date:
        st.sidebar.error("End date must be after start date.")
        return
    
    strategy = st.sidebar.selectbox('Choose a Trading Strategy', 
                                    ['Moving Average Crossover', 'RSI', 'Bollinger Bands'])
    
    # Strategy descriptions
    strategy_descriptions = {
        'Moving Average Crossover': 'This strategy identify potential buy and sell signals. It works by comparing two moving averages: a short-term and a long-term. When the short-term average crosses above the long-term average, it indicates a bullish trend, suggesting it is a good time to buy. Conversely, when the short-term average crosses below, it signals a bearish trend, suggesting a sell or hold position.',
        'RSI': 'The Relative Strength Index (RSI) identify potential buy and sell signals based on momentum. It measures the speed and change of price movements, with values ranging from 0 to 100. An RSI above 70 indicates that an asset may be overbought, signaling a potential sell opportunity. Conversely, an RSI below 30 suggests the asset is oversold, presenting a potential buy opportunity. Monitoring these levels helps gauge market conditions effectively.',
        'Bollinger Bands': 'Bollinger Bands identify buy and sell signals based on price movements relative to three bands: a middle band (simple moving average), an upper band, and a lower band, calculated using standard deviations. A buy signal occurs when the price falls below the lower band, indicating possible overselling and a potential price bounce back. Conversely, a sell signal is generated when the price rises above the upper band, suggesting overbuying and a likely price correction.'
    }
    
    st.sidebar.info(strategy_descriptions[strategy])
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Loading data with a spinner
        with st.spinner('Loading data...'):
            data = load_data(ticker, start_date - timedelta(days=30), end_date)  # Load extra 30 days for strategy initialization
        
        if data.empty:
            st.error("No data found for the selected ticker and date range.")
            return
        
        # Execute selected strategy
        if strategy == 'Moving Average Crossover':
            signals = execute_moving_average_crossover_strategy(data)
        elif strategy == 'RSI':
            signals = execute_rsi_strategy(data)
        elif strategy == 'Bollinger Bands':
            signals = execute_bollinger_bands_strategy(data)
        
        # Trim signals to match user-selected date range
        signals = signals.loc[start_date:end_date]
        
        # Plot results
        fig = plot_strategy_results(data.loc[start_date:end_date], signals, strategy)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader('Strategy Performance')
        
        if 'cumulative_strategy_returns' in signals.columns:
            metrics = calculate_metrics(signals)
            
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("Initial Investment", f"${metrics['initial_investment']:,.2f}")
                st.metric("Final Portfolio Value", f"${metrics['final_value']:,.2f}")
                st.metric("Total Return", f"{metrics['total_return']:.2f}%")
            with col2b:
                st.metric("Number of Trades", metrics['num_trades'])
                st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
            
            # Display a sample of the signals dataframe in a better format
            st.subheader("Signal Data (Sample)")
            st.write("Below are the last few entries of the signal data:")
            st.dataframe(signals[['price', 'positions', 'cumulative_strategy_returns', 'cumulative_returns']].tail(), use_container_width=True)
            
            # Download results
            csv = signals.to_csv().encode('utf-8')
            st.download_button("Download Full Strategy Results (CSV)", csv, "strategy_results.csv", "text/csv")
        else:
            st.warning("Strategy performance data not available.")
    
    # Additional insights and explanations
    st.subheader("Strategy Insights")
    st.write(f"The {strategy} strategy was applied to {ticker} stock from {start_date} to {end_date}.")
    st.write(strategy_descriptions[strategy])
    st.write("Key observations:")
    st.write("- The green triangles on the chart indicate buy signals.")
    st.write("- The red triangles on the chart indicate sell signals.")
    st.write("- The strategy's performance is based on these signals and the stock's price movements.")
    # st.write("- The bottom chart compares the cumulative returns of the strategy (purple) to the market returns (orange).")
    
    st.subheader("Performance Metrics Explained")
    st.write("- **Total Return**: The overall percentage gain or loss from the strategy.")
    st.write("- **Number of Trades**: The total number of buy and sell transactions executed.")
    # st.write("- **Win Rate**: The percentage of trades that resulted in a profit.")
    st.write("- **Sharpe Ratio**: A measure of risk-adjusted return. A higher Sharpe ratio indicates better risk-adjusted performance.")
    st.write("- **Max Drawdown**: The maximum observed loss from a peak to a trough, before a new peak is attained. It's a measure of downside risk.")
    
    st.info("Note: Past performance does not guarantee future results. This tool is for educational purposes only and should not be considered as financial advice.")

if __name__ == '__main__':
    main()