# tests/test_ml_strategy.py

import unittest
import pandas as pd
import numpy as np
from strategies.ml_strategy import ml_strategy

class TestMLStrategy(unittest.TestCase):
    def setUp(self):
        # Create sample data with 100 rows
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=100)
        close_prices = np.random.randint(100, 200, size=100)
        self.data = pd.DataFrame({
            'Close': close_prices,
            'Open': close_prices + np.random.randint(-5, 5, size=100),
            'High': close_prices + np.random.randint(0, 10, size=100),
            'Low': close_prices - np.random.randint(0, 10, size=100),
            'Volume': np.random.randint(1000, 2000, size=100)
        }, index=dates)

    def test_ml_strategy(self):
        try:
            results = ml_strategy(self.data)
            self.assertIn('Signal', results.columns)
            self.assertIn('Position', results.columns)
            self.assertIn('Strategy_Returns', results.columns)
            self.assertLess(len(results), len(self.data))  # Results should be shorter due to train-test split
        except Exception as e:
            self.fail(f"ml_strategy raised an exception: {str(e)}")

    def test_insufficient_data(self):
        with self.assertRaises(ValueError):
            ml_strategy(self.data.iloc[:10])  # Try with only 10 data points

if __name__ == '__main__':
    unittest.main()