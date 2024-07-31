import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, QTextEdit, QMessageBox, QProgressBar, QSplitter
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon
import pandas as pd
import numpy as np
import yfinance as yf
from backtesting.backtest import Backtest
from strategies.moving_average_crossover import moving_average_crossover
from strategies.rsi_strategy import rsi_strategy
from strategies.bollinger_bands import bollinger_bands
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from datetime import datetime, timedelta
from config import MOVING_AVERAGE_SHORT_WINDOW, MOVING_AVERAGE_LONG_WINDOW, RSI_WINDOW, RSI_OVERBOUGHT, RSI_OVERSOLD, BOLLINGER_WINDOW, BOLLINGER_NUM_STD

class BacktestThread(QThread):
    finished = pyqtSignal(object, object, object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, symbol, start_date, end_date, strategy):
        QThread.__init__(self)
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.strategy = strategy

    def run(self):
        try:
            self.progress.emit(10)
            data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
            if data.empty:
                self.error.emit(f"No data available for {self.symbol} in the specified date range.")
                return
            
            self.progress.emit(40)
            if self.strategy == "Moving Average Crossover":
                strategy_func = moving_average_crossover
            elif self.strategy == "RSI Strategy":
                strategy_func = rsi_strategy
            elif self.strategy == "Bollinger Bands":
                strategy_func = bollinger_bands

            self.progress.emit(60)
            backtest = Backtest(data, strategy_func)
            results = backtest.run()
            metrics = backtest.get_performance_metrics()
            
            self.progress.emit(100)
            self.finished.emit(data, results, metrics)
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quantitative Trading Backtester")
        self.setGeometry(100, 100, 1200, 800)
        # self.setWindowIcon(QIcon('icon.png'))

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Input panel
        input_panel = QWidget()
        input_layout = QHBoxLayout()
        input_panel.setLayout(input_layout)

        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("Enter stock symbol (e.g., AAPL)")
        input_layout.addWidget(QLabel("Stock Symbol:"))
        input_layout.addWidget(self.symbol_input)

        self.start_date_input = QLineEdit()
        self.start_date_input.setPlaceholderText("YYYY-MM-DD")
        input_layout.addWidget(QLabel("Start Date:"))
        input_layout.addWidget(self.start_date_input)

        self.end_date_input = QLineEdit()
        self.end_date_input.setPlaceholderText("YYYY-MM-DD")
        input_layout.addWidget(QLabel("End Date:"))
        input_layout.addWidget(self.end_date_input)

        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["Moving Average Crossover", "RSI Strategy", "Bollinger Bands"])
        input_layout.addWidget(QLabel("Strategy:"))
        input_layout.addWidget(self.strategy_combo)

        self.run_button = QPushButton("Run Backtest")
        self.run_button.clicked.connect(self.run_backtest)
        input_layout.addWidget(self.run_button)

        main_layout.addWidget(input_panel)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        # Splitter for results and chart
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter, 1)

        # Results display
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        splitter.addWidget(self.results_text)

        # Chart display
        chart_widget = QWidget()
        chart_layout = QVBoxLayout()
        chart_widget.setLayout(chart_layout)
        self.figure, self.ax = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
        self.canvas = FigureCanvas(self.figure)
        chart_layout.addWidget(self.canvas)
        splitter.addWidget(chart_widget)

    def run_backtest(self):
        symbol = self.symbol_input.text()
        if not symbol:
            self.show_error("Please enter a stock symbol.")
            return

        start_date = self.start_date_input.text() or (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        end_date = self.end_date_input.text() or datetime.now().strftime('%Y-%m-%d')

        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            self.show_error("Invalid date format. Please use YYYY-MM-DD.")
            return

        strategy = self.strategy_combo.currentText()

        self.run_button.setEnabled(False)
        self.progress_bar.setValue(0)

        self.backtest_thread = BacktestThread(symbol, start_date, end_date, strategy)
        self.backtest_thread.finished.connect(self.on_backtest_finished)
        self.backtest_thread.error.connect(self.show_error)
        self.backtest_thread.progress.connect(self.update_progress)
        self.backtest_thread.start()

    def on_backtest_finished(self, data, results, metrics):
        self.run_button.setEnabled(True)
        self.progress_bar.setValue(100)

        # Display results
        self.results_text.clear()
        self.results_text.append(f"Strategy: {self.strategy_combo.currentText()}")
        self.results_text.append(f"Symbol: {self.symbol_input.text()}")
        self.results_text.append(f"Period: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        self.results_text.append("\nPerformance Metrics:")
        for key, value in metrics.items():
            self.results_text.append(f"{key}: {value}")

        # Plot results
        self.plot_results(data, results, self.strategy_combo.currentText())

    def plot_results(self, data, results, strategy):
        self.ax[0].clear()
        self.ax[1].clear()

        # Plot stock price and portfolio value
        self.ax[0].plot(data.index, data['Close'], label='Stock Price', alpha=0.7)
        self.ax[0].plot(results.index, results['total'], label='Portfolio Value', alpha=0.7)

        # Plot buy and sell signals if available
        if 'signal' in results.columns:
            buy_signals = results[results['signal'] == 1]
            sell_signals = results[results['signal'] == -1]
            self.ax[0].scatter(buy_signals.index, data.loc[buy_signals.index, 'Close'], marker='^', color='g', label='Buy Signal')
            self.ax[0].scatter(sell_signals.index, data.loc[sell_signals.index, 'Close'], marker='v', color='r', label='Sell Signal')

        # Plot strategy-specific indicators
        if strategy == "Moving Average Crossover":
            short_ma = data['Close'].rolling(window=MOVING_AVERAGE_SHORT_WINDOW).mean()
            long_ma = data['Close'].rolling(window=MOVING_AVERAGE_LONG_WINDOW).mean()
            self.ax[0].plot(short_ma.index, short_ma, label=f'{MOVING_AVERAGE_SHORT_WINDOW}-day MA', alpha=0.7)
            self.ax[0].plot(long_ma.index, long_ma, label=f'{MOVING_AVERAGE_LONG_WINDOW}-day MA', alpha=0.7)
        elif strategy == "RSI Strategy":
            if 'rsi' in results.columns:
                rsi = results['rsi']
                self.ax[1].plot(rsi.index, rsi, label='RSI', color='purple')
                self.ax[1].axhline(y=RSI_OVERBOUGHT, color='r', linestyle='--', alpha=0.7)
                self.ax[1].axhline(y=RSI_OVERSOLD, color='g', linestyle='--', alpha=0.7)
                self.ax[1].set_ylim(0, 100)
                self.ax[1].set_ylabel('RSI')
        elif strategy == "Bollinger Bands":
            if 'upper' in results.columns and 'lower' in results.columns and 'middle' in results.columns:
                self.ax[0].plot(results['upper'], label='Upper Band', linestyle='--', alpha=0.7)
                self.ax[0].plot(results['lower'], label='Lower Band', linestyle='--', alpha=0.7)
                self.ax[0].plot(results['middle'], label='Middle Band', linestyle='--', alpha=0.7)

        self.ax[0].set_title(f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}: {strategy}")
        self.ax[0].set_xlabel("Date")
        self.ax[0].set_ylabel("Price")
        self.ax[0].legend()

        if strategy != "RSI Strategy" and 'positions' in results.columns:
            self.ax[1].plot(results['positions'], label='Position', color='orange')
            self.ax[1].set_ylabel('Position')

        self.ax[1].set_xlabel("Date")
        self.ax[1].legend()

        plt.tight_layout()
        self.canvas.draw()

    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)
        self.run_button.setEnabled(True)
        self.progress_bar.setValue(0)

    def update_progress(self, value):
        self.progress_bar.setValue(value)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())