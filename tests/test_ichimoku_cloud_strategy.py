# tests/test_ichimoku_cloud_strategy.py

import unittest
import pandas as pd
import numpy as np
from strategies.ichimoku_cloud_strategy import calculate_ichimoku_cloud, ichimoku_cloud_strategy

class TestIchimokuCloudStrategy(unittest.TestCase):
    def setUp(self):
        # Create sample data
        np.random.seed(42)  # Set seed for reproducibility
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
        prices = np.random.randn(len(dates)).cumsum() + 100
        self.sample_data = pd.DataFrame({
            'close': prices,
            'high': prices + np.random.rand(len(dates)),
            'low': prices - np.random.rand(len(dates))
        }, index=dates)

    def test_calculate_ichimoku_cloud(self):
        ichimoku = calculate_ichimoku_cloud(self.sample_data)
        
        self.assertIsInstance(ichimoku, pd.DataFrame)
        self.assertIn('conversion_line', ichimoku.columns)
        self.assertIn('base_line', ichimoku.columns)
        self.assertIn('leading_span_a', ichimoku.columns)
        self.assertIn('leading_span_b', ichimoku.columns)
        self.assertIn('lagging_span', ichimoku.columns)
        self.assertEqual(len(ichimoku), len(self.sample_data))

    def test_ichimoku_cloud_strategy(self):
        signals = ichimoku_cloud_strategy(self.sample_data)
        
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('signal', signals.columns)
        self.assertIn('positions', signals.columns)
        self.assertIn('conversion_line', signals.columns)
        self.assertIn('base_line', signals.columns)
        self.assertIn('leading_span_a', signals.columns)
        self.assertIn('leading_span_b', signals.columns)
        
        # Check if signals are either -1, 0, or 1
        self.assertTrue(set(signals['signal'].unique()).issubset({-1, 0, 1}))
        
        # Check if positions are -2, -1, 0, 1, or 2
        self.assertTrue(set(signals['positions'].unique()).issubset({-2, -1, 0, 1, 2}))

if __name__ == '__main__':
    unittest.main()