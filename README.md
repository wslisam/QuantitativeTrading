# QuantitativeTrading
Quantitative Trading 

Tool for Algorithmic trading strategy visualization

## Features

- Multiple trading strategies (Moving Average Crossover, RSI , Bollinger_bands)
- Backtesting engine with performance metrics
- Interactive GUI for strategy selection and visualization
- Risk management utilities
- Portfolio optimization

## Installation

1. Clone the repository:
git clone https://github.com/wslisam/QuantitativeTrading.git
cd QuantitativeTrading
pip install -r requirements.txt

2. Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate # On Windows, use: venv\Scripts\activate


3. Install the required packages:
pip install -r requirements.txt


## Usage

Run the main application:

streamlit run main.py

You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501

For better performance, install the Watchdog module:
$ xcode-select --install
$ pip install watchdog

This will launch the GUI where you can:

1. **Enter a Stock Symbol**: Input the ticker symbol of the stock you wish to analyze (e.g., AAPL, TSLA).
2. **Select a Trading Strategy**: Choose from various available strategies, including:
   - Moving Average Crossover
   - RSI (Relative Strength Index)
   - Bollinger Bands
3. **Configure Parameters**: Adjust strategy parameters as needed to customize your backtest.
4. **Run a Backtest**: Execute the strategy over the selected date range and analyze its performance.
5. **View Performance Metrics**: Access key performance indicators such as total return and final portfolio value.
6. **Visualize Results**: Interactively explore charts displaying price movements, buy/sell signals, and trading volume.

## Running Tests

To run the tests, use the following command:

pytest tests/

To generate a coverage report:
pytest --cov=strageies --cov-report=html tests/

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).




```
Implement best practices
- Use type hints
- Follow PEP 8 style guide
- Write docstrings for all functions and classes
- Use meaningful variable and function names
```