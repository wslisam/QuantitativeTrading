import pandas as pd
import matplotlib.pyplot as plt

class Backtest:
    def __init__(self, data, strategy, initial_capital=100000.0):
        self.data = data
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.positions = None
        self.portfolio = None

    def run(self):
        signals = self.strategy(self.data)
        self.positions = pd.DataFrame(index=signals.index).fillna(0.0)
        self.positions['Stock'] = 100 * signals['signal']
        
        self.portfolio = self.positions.multiply(self.data['Close'], axis=0)
        pos_diff = self.positions.diff()
        
        self.portfolio['holdings'] = (self.positions.multiply(self.data['Close'], axis=0)).sum(axis=1)
        self.portfolio['cash'] = self.initial_capital - (pos_diff.multiply(self.data['Close'], axis=0)).sum(axis=1).cumsum()
        
        self.portfolio['total'] = self.portfolio['cash'] + self.portfolio['holdings']
        self.portfolio['returns'] = self.portfolio['total'].pct_change()
        
        return self.portfolio

    def plot_results(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
        
        self.data['Close'].plot(ax=ax1)
        ax1.set_ylabel('Price')
        ax1.set_title('Stock Price and Trading Signals')
        
        ax1.plot(self.data.index, self.data['Close'], label='Close Price')
        ax1.plot(self.positions.loc[self.positions['Stock'] == 100].index, 
                 self.data['Close'][self.positions['Stock'] == 100],
                 '^', markersize=10, color='g', label='Buy Signal')
        ax1.plot(self.positions.loc[self.positions['Stock'] == -100].index, 
                 self.data['Close'][self.positions['Stock'] == -100],
                 'v', markersize=10, color='r', label='Sell Signal')
        ax1.legend()
        
        self.portfolio['total'].plot(ax=ax2)
        ax2.set_ylabel('Portfolio Value')
        ax2.set_title('Portfolio Value over Time')
        
        plt.tight_layout()
        plt.show()

    def get_performance_metrics(self):
        total_return = (self.portfolio['total'].iloc[-1] - self.initial_capital) / self.initial_capital
        sharpe_ratio = self.portfolio['returns'].mean() / self.portfolio['returns'].std() * (252 ** 0.5)  # Assuming 252 trading days
        max_drawdown = (self.portfolio['total'] / self.portfolio['total'].cummax() - 1).min()
        
        return {
            'Total Return': f'{total_return:.2%}',
            'Sharpe Ratio': f'{sharpe_ratio:.2f}',
            'Max Drawdown': f'{max_drawdown:.2%}'
        }