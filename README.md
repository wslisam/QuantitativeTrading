# QuantitativeTrading
Quantitative Trading 

An algorithmic trading system.

## Features

- Multiple trading strategies (Moving Average Crossover, RSI)
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
1. Enter a stock symbol
2. Select a trading strategy
3. Run a backtest
4. View performance metrics and charts

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