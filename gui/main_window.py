import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QComboBox, QTextEdit, QMessageBox, QProgressBar, 
                             QSplitter, QTabWidget, QGridLayout, QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon
import pandas as pd
import numpy as np
import yfinance as yf
from backtesting.backtest import Backtest
from strategies import *
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
            elif self.strategy == "Machine Learning Strategy": 
                strategy_func = ml_strategy

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
        self.setGeometry(100, 100, 1400, 900)
        # self.setWindowIcon(QIcon('icon.png'))

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Input panel
        input_panel = QWidget()
        input_layout = QGridLayout()
        input_panel.setLayout(input_layout)

        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("Enter stock symbol (e.g., AAPL)")
        input_layout.addWidget(QLabel("Stock Symbol:"), 0, 0)
        input_layout.addWidget(self.symbol_input, 0, 1)

        self.start_date_input = QLineEdit()
        self.start_date_input.setPlaceholderText("YYYY-MM-DD")
        input_layout.addWidget(QLabel("Start Date:"), 0, 2)
        input_layout.addWidget(self.start_date_input, 0, 3)

        self.end_date_input = QLineEdit()
        self.end_date_input.setPlaceholderText("YYYY-MM-DD")
        input_layout.addWidget(QLabel("End Date:"), 0, 4)
        input_layout.addWidget(self.end_date_input, 0, 5)

        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["Moving Average Crossover", "RSI Strategy", "Bollinger Bands", "Machine Learning Strategy"])
        input_layout.addWidget(QLabel("Strategy:"), 1, 0)
        input_layout.addWidget(self.strategy_combo, 1, 1)

        self.run_button = QPushButton("Run Backtest")
        self.run_button.clicked.connect(self.run_backtest)
        input_layout.addWidget(self.run_button, 1, 5)

        main_layout.addWidget(input_panel)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        # Tab widget for results and charts
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Results tab
        results_tab = QWidget()
        results_layout = QVBoxLayout()
        results_tab.setLayout(results_layout)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        results_layout.addWidget(self.results_table)

        self.tab_widget.addTab(results_tab, "Results")

        # Charts tab
        charts_tab = QWidget()
        charts_layout = QVBoxLayout()
        charts_tab.setLayout(charts_layout)

        self.figure, (self.ax1, self.ax2, self.ax3, self.ax4) = plt.subplots(4, 1, figsize=(12, 20), gridspec_kw={'height_ratios': [3, 1, 1, 1]})
        self.canvas = FigureCanvas(self.figure)
        charts_layout.addWidget(self.canvas)

        self.tab_widget.addTab(charts_tab, "Charts")

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

        # Display results in table
        self.results_table.setRowCount(len(metrics) + 3)
        self.results_table.setItem(0, 0, QTableWidgetItem("Strategy"))
        self.results_table.setItem(0, 1, QTableWidgetItem(self.strategy_combo.currentText()))
        self.results_table.setItem(1, 0, QTableWidgetItem("Symbol"))
        self.results_table.setItem(1, 1, QTableWidgetItem(self.symbol_input.text()))
        self.results_table.setItem(2, 0, QTableWidgetItem("Period"))
        self.results_table.setItem(2, 1, QTableWidgetItem(f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}"))

        for i, (key, value) in enumerate(metrics.items(), start=3):
            self.results_table.setItem(i, 0, QTableWidgetItem(key))
            self.results_table.setItem(i, 1, QTableWidgetItem(str(value)))

        # Plot results
        self.plot_results(data, results, self.strategy_combo.currentText())

        # Switch to Results tab
        self.tab_widget.setCurrentIndex(0)

    def plot_results(self, data, results, strategy):
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()

        # Plot stock price and portfolio value
        self.ax1.plot(data.index, data['Close'], label='Stock Price', alpha=0.7)
        self.ax1.plot(results.index, results['total'], label='Portfolio Value', alpha=0.7)

        # Plot buy and sell signals if available
        if 'signal' in results.columns:
            buy_signals = results[results['signal'] == 1]
            sell_signals = results[results['signal'] == -1]
            self.ax1.scatter(buy_signals.index, data.loc[buy_signals.index, 'Close'], marker='^', color='g', s=100, label='Buy Signal')
            self.ax1.scatter(sell_signals.index, data.loc[sell_signals.index, 'Close'], marker='v', color='r', s=100, label='Sell Signal')

        # Plot strategy-specific indicators
        if strategy == "Machine Learning Strategy":
            if 'Signal' in results.columns:
                buy_signals = results[results['Signal'] == 1]
                sell_signals = results[results['Signal'] == 0]
                self.ax1.scatter(buy_signals.index, data.loc[buy_signals.index, 'Close'], marker='^', color='g', s=100, label='Buy Signal')
                self.ax1.scatter(sell_signals.index, data.loc[sell_signals.index, 'Close'], marker='v', color='r', s=100, label='Sell Signal')
        elif strategy == "Moving Average Crossover":
            short_ma = data['Close'].rolling(window=MOVING_AVERAGE_SHORT_WINDOW).mean()
            long_ma = data['Close'].rolling(window=MOVING_AVERAGE_LONG_WINDOW).mean()
            self.ax1.plot(short_ma.index, short_ma, label=f'{MOVING_AVERAGE_SHORT_WINDOW}-day MA', alpha=0.7)
            self.ax1.plot(long_ma.index, long_ma, label=f'{MOVING_AVERAGE_LONG_WINDOW}-day MA', alpha=0.7)
        elif strategy == "RSI Strategy":
            if 'rsi' in results.columns:
                rsi = results['rsi']
                self.ax2.plot(rsi.index, rsi, label='RSI', color='purple')
                self.ax2.axhline(y=RSI_OVERBOUGHT, color='r', linestyle='--', alpha=0.7)
                self.ax2.axhline(y=RSI_OVERSOLD, color='g', linestyle='--', alpha=0.7)
                self.ax2.set_ylim(0, 100)
                self.ax2.set_ylabel('RSI')
        elif strategy == "Bollinger Bands":
            if 'upper' in results.columns and 'lower' in results.columns and 'middle' in results.columns:
                self.ax1.plot(results.index, results['upper'], label='Upper Band', linestyle='--', alpha=0.7)
                self.ax1.plot(results.index, results['lower'], label='Lower Band', linestyle='--', alpha=0.7)
                self.ax1.plot(results.index, results['middle'], label='Middle Band', linestyle='--', alpha=0.7)

        self.ax1.set_title(f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}: {strategy}", fontsize=16)
        self.ax1.set_ylabel("Price", fontsize=12)
        self.ax1.legend(fontsize=10)

        # Plot volume
        self.ax3.bar(data.index, data['Volume'], label='Volume', alpha=0.7)
        self.ax3.set_ylabel("Volume", fontsize=12)

        # Plot returns
        returns = data['Close'].pct_change()
        self.ax4.plot(returns.index, returns, label='Daily Returns', alpha=0.7)
        self.ax4.set_ylabel("Returns", fontsize=12)

        # Plot positions if available
        if 'positions' in results.columns:
            self.ax2.plot(results.index, results['positions'], label='Position', color='orange')
            self.ax2.set_ylabel('Position', fontsize=12)

        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.set_xticklabels([])

        self.ax4.set_xlabel("Date", fontsize=12)

        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.legend(fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.tick_params(axis='both', which='major', labelsize=10)

        # Format y-axis labels to avoid scientific notation
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

        # Adjust date formatting on x-axis
        self.figure.autofmt_xdate()

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