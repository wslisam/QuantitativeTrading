import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import yfinance as yf
import pandas as pd
import numpy as np
from strategies import *
from datetime import timedelta
from utils.portfolio_optimization import *
from config import *
from backtesting.backtest import Backtest
from utils.data_fetcher import fetch_data
from utils.risk_management import calculate_position_size, trailing_stop_loss 
import time

# Strategy descriptions
strategy_descriptions = {
    'Moving Average Crossover': 'This strategy identify potential buy and sell signals. It works by comparing two moving averages: a short-term and a long-term. When the short-term average crosses above the long-term average, it indicates a bullish trend, suggesting it is a good time to buy. Conversely, when the short-term average crosses below, it signals a bearish trend, suggesting a sell or hold position.',
    'RSI': 'The Relative Strength Index (RSI) identify potential buy and sell signals based on momentum. It measures the speed and change of price movements, with values ranging from 0 to 100. An RSI above 70 indicates that an asset may be overbought, signaling a potential sell opportunity. Conversely, an RSI below 30 suggests the asset is oversold, presenting a potential buy opportunity. Monitoring these levels helps gauge market conditions effectively.',
    'Bollinger Bands': 'Bollinger Bands identify buy and sell signals based on price movements relative to three bands: a middle band (simple moving average), an upper band, and a lower band, calculated using standard deviations. A buy signal occurs when the price falls below the lower band, indicating possible overselling and a potential price bounce back. Conversely, a sell signal is generated when the price rises above the upper band, suggesting overbuying and a likely price correction.'
}
    
@st.cache_data  # the caching decorator
# Function to plot strategy results
def plot_strategy_results(data, signals, strategy_name):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.12, 
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
    
    fig.update_layout(
        height=1000,  # Increased height
        margin=dict(l=50, r=50, t=100, b=50),  # Reduced top margin
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        hovermode="x unified"
    )
    
    # Make the layout more responsive
    fig.update_layout(
        autosize=True,
        width=None,
    )
    
    # Adjust the title
    fig.update_layout(
        title=dict(
            text=f"{strategy_name}<br>Strategy Results",  # Line break in title
            y=0.97,
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(size=20)  # Reduced font size
        )
    )
    # Ensure x-axis labels are visible and not cut off
    fig.update_xaxes(showticklabels=True, tickangle=45)
    
    # Adjust y-axis to prevent cutting off labels
    fig.update_yaxes(automargin=True)
    
    # Add more space between subplots
    fig.update_layout(height=1000)
    
    # Increase font size for better readability
    fig.update_layout(
        font=dict(size=14),
        xaxis_title_font=dict(size=16),
        yaxis_title_font=dict(size=16)
    )
    
    # Add axis labels
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="Returns", row=3, col=1)
    
    return fig

def risk_management_section(data):
    st.header("Risk Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        portfolio_value = st.number_input("Portfolio Value ($)", min_value=1000, value=10000, step=1000)
        risk_per_trade = st.slider("Risk per Trade (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.1) / 100
    
    with col2:
        stop_loss_percent = st.slider("Stop Loss (%)", min_value=1.0, max_value=10.0, value=2.0, step=0.1) / 100
        trailing_percent = st.slider("Trailing Stop (%)", min_value=1.0, max_value=10.0, value=1.5, step=0.1) / 100
    
    # Calculate position size
    position_size = calculate_position_size(portfolio_value, risk_per_trade, stop_loss_percent)
    max_loss = portfolio_value * risk_per_trade
    
    col1, col2 = st.columns(2)
    col1.metric("Recommended Position Size ($)", f"{position_size:.2f}")
    col2.metric("Maximum Loss per Trade ($)", f"{max_loss:.2f}")
    
    if not data.empty:
        entry_price = data['Close'].iloc[-1]  # Use the last closing price as entry price
        current_stop_loss = trailing_stop_loss(data, entry_price, stop_loss_percent, trailing_percent)
        potential_loss = (entry_price - current_stop_loss) * (position_size / entry_price)
        
        col1, col2 = st.columns(2)
        col1.metric("Current Trailing Stop Loss ($)", f"{current_stop_loss:.2f}")
        col2.metric("Potential Loss ($)", f"{potential_loss:.2f}")
    else:
        st.warning("Please load stock data to calculate trailing stop loss.")
    
    st.info("""
    Risk Management Explanation:
    - Position Size: The amount you should invest based on your risk tolerance.
    - Maximum Loss: The maximum amount you're willing to lose on this trade.
    - Trailing Stop Loss: A dynamic stop loss that adjusts as the price moves in your favor.
    - Potential Loss: The estimated loss if the stop loss is triggered.
    
    Always ensure that your potential loss aligns with your risk tolerance.
    """)
    
def optimize_and_plot_portfolio(tickers, start_date, end_date, strategy, min_weight=0.05, max_weight=0.4, min_assets=3, risk_free_rate=0.02):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    returns = data.pct_change().dropna()

    weights = safe_optimize_portfolio(returns, strategy, min_weight, max_weight, min_assets, risk_free_rate)

    portfolio_return, portfolio_volatility = portfolio_performance(weights, returns)

    if strategy in ['sharpe', 'sortino']:
        performance_metric = (portfolio_return - risk_free_rate) / portfolio_volatility
        metric_name = f"{strategy.capitalize()} Ratio"
    elif strategy == 'max_return':
        performance_metric = portfolio_return
        metric_name = "Expected Annual Return"
    elif strategy == 'min_volatility':
        performance_metric = portfolio_volatility
        metric_name = "Expected Annual Volatility"

    fig = go.Figure(data=[go.Pie(labels=tickers, values=weights, textinfo='label+percent', hole=.3)])
    fig.update_layout(title_text="Optimal Portfolio Weights")

    asset_returns = returns.mean() * 252
    asset_volatilities = returns.std() * np.sqrt(252)

    scatter_fig = go.Figure()
    scatter_fig.add_trace(go.Scatter(
        x=asset_volatilities,
        y=asset_returns,
        mode='markers+text',
        marker=dict(size=10, color=asset_returns, colorscale='Viridis', showscale=True),
        text=tickers,
        textposition="top center"
    ))
    scatter_fig.add_trace(go.Scatter(
        x=[portfolio_volatility],
        y=[portfolio_return],
        mode='markers+text',
        marker=dict(size=15, color='red', symbol='star'),
        text=['Optimized Portfolio'],
        textposition="top center"
    ))
    scatter_fig.update_layout(
        title='Asset Risk-Return Profile',
        xaxis_title='Volatility (Risk)',
        yaxis_title='Expected Return',
        showlegend=False
    )

    return fig, scatter_fig, weights, portfolio_return, portfolio_volatility, performance_metric, metric_name

def strategy_configuration_sidebar():
    # Sidebar for user input
    st.sidebar.header('Strategy Configuration')
    ticker = st.sidebar.text_input('Enter Stock Ticker', 'AAPL', help="Enter a valid stock ticker symbol (e.g., AAPL, TSLA).")
    
    start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2020-01-01'))
    end_date = st.sidebar.date_input('End Date', pd.to_datetime('2023-01-01'))

    if end_date < start_date:
        st.sidebar.error("End date must be after start date.")
        return
    
    strategy = st.sidebar.selectbox("Choose a Trading Strategy", list(strategy_descriptions.keys()))
    
    st.sidebar.info(strategy_descriptions[strategy])
    return ticker, start_date, end_date, strategy

def main():
    st.set_page_config(layout="wide", page_title="Trading Strategy Visualization")
    st.title('Trading Strategy Visualization')
    
    ticker, start_date, end_date, strategy = strategy_configuration_sidebar()

    # Main content area
    with st.spinner('Loading data...'):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        data = fetch_data(ticker, start_date - timedelta(days=30), end_date)
    
    if data.empty:
        st.error("No data found for the selected ticker and date range.")
        return
    
    if strategy == 'Moving Average Crossover':
        backtest = Backtest(data, execute_moving_average_crossover_strategy)
    elif strategy == 'RSI':
        backtest = Backtest(data, execute_rsi_strategy)
    elif strategy == 'Bollinger Bands':
        backtest = Backtest(data, execute_bollinger_bands_strategy)
    
    signals = backtest.run()
    signals = signals.loc[start_date:end_date]
    
    fig = plot_strategy_results(data.loc[start_date:end_date], signals, strategy)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader('Strategy Performance')
    
    if 'cumulative_strategy_returns' in signals.columns:
        metrics = backtest.calculate_metrics(signals)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Initial Investment", f"${metrics['initial_investment']:,.2f}")
        col2.metric("Final Portfolio Value", f"${metrics['final_value']:,.2f}")
        col3.metric("Total Return", f"{metrics['total_return']:.2f}%")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Number of Trades", metrics['num_trades'])
        col2.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        col3.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
        
        st.subheader("Signal Data (Sample)")
        st.dataframe(signals[['price', 'positions', 'cumulative_strategy_returns', 'cumulative_returns']].tail(), use_container_width=True)
        
        csv = signals.to_csv().encode('utf-8')
        st.download_button("Download Full Strategy Results (CSV)", csv, "strategy_results.csv", "text/csv")
    else:
        st.warning("Strategy performance data not available.")
    
    st.subheader("Strategy Insights")
    st.write(f"The {strategy} strategy was applied to {ticker} stock from {start_date} to {end_date}.")
    st.write(strategy_descriptions[strategy])
    st.write("Key observations:")
    st.write("- The green triangles on the chart indicate buy signals.")
    st.write("- The red triangles on the chart indicate sell signals.")
    st.write("- The strategy's performance is based on these signals and the stock's price movements.")
    
    st.subheader("Performance Metrics Explained")
    st.write("- **Total Return**: The overall percentage gain or loss from the strategy.")
    st.write("- **Number of Trades**: The total number of buy and sell transactions executed.")
    st.write("- **Sharpe Ratio**: A measure of risk-adjusted return. A higher Sharpe ratio indicates better risk-adjusted performance.")
    st.write("- **Max Drawdown**: The maximum observed loss from a peak to a trough, before a new peak is attained. It's a measure of downside risk.")
    
    st.info("Note: Past performance does not guarantee future results. This tool is for educational purposes only and should not be considered as financial advice.")

    # Portfolio Optimization Section
    st.header("Portfolio Optimization")
    st.write("Optimize a portfolio by selecting multiple stocks:")

    portfolio_tickers = st.multiselect("Select stocks for portfolio (3-10 recommended)", 
                                    ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V'],
                                    default=['AAPL', 'GOOGL', 'MSFT'])

    if len(portfolio_tickers) >= 3:
        col1, col2, col3 = st.columns(3)
        with col1:
            min_weight = st.slider("Minimum weight per asset", 0.01, 0.1, 0.05, 0.01)
        with col2:
            max_weight = st.slider("Maximum weight per asset", 0.2, 0.5, 0.4, 0.05)
        with col3:
            min_assets = st.slider("Minimum number of assets", 2, len(portfolio_tickers), 3, 1)
        
        longer_period = st.checkbox("Use 5-year historical data for optimization")
        opt_start_date = start_date - pd.DateOffset(years=4) if longer_period else start_date
        
        risk_free_rate = st.slider("Risk-free rate (%)", 0.0, 5.0, 2.0, 0.1) / 100

        optimization_strategy = st.selectbox(
            "Select optimization strategy",
            ["Sharpe Ratio", "Sortino Ratio", "Maximum Return", "Minimum Volatility"],
            index=0
        )

        strategy_mapping = {
            "Sharpe Ratio": "sharpe",
            "Sortino Ratio": "sortino",
            "Maximum Return": "max_return",
            "Minimum Volatility": "min_volatility"
        }

        with st.spinner('Optimizing portfolio...'):
            pie_fig, scatter_fig, weights, portfolio_return, portfolio_volatility, performance_metric, metric_name = optimize_and_plot_portfolio(
                portfolio_tickers, opt_start_date, end_date, strategy_mapping[optimization_strategy],
                min_weight, max_weight, min_assets, risk_free_rate
            )
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(pie_fig, use_container_width=True)
        with col2:
            st.plotly_chart(scatter_fig, use_container_width=True)
        
        st.subheader("Optimization Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("Expected Annual Return", f"{portfolio_return:.2%}")
        col2.metric("Expected Annual Volatility", f"{portfolio_volatility:.2%}")
        col3.metric(metric_name, f"{performance_metric:.2f}")
        
        st.subheader("Optimal Portfolio Weights")
        weight_df = pd.DataFrame({'Stock': portfolio_tickers, 'Weight': weights})
        weight_df = weight_df.sort_values('Weight', ascending=False).reset_index(drop=True)
        st.dataframe(weight_df.style.format({'Weight': '{:.2%}'}), use_container_width=True)
        
        csv = weight_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Optimal Weights (CSV)", csv, "optimal_portfolio_weights.csv", "text/csv")
        
        st.info(f"This optimization aims to {optimization_strategy.lower()}. The results show the optimal allocation of your investment across the selected stocks based on historical data.")
        
    else:
        st.warning("Please select at least three stocks for portfolio optimization.")
    
    risk_management_section(data)

if __name__ == '__main__':
    main()