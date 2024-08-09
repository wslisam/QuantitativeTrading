# tests/test_stochastic_oscillator_strategy.py

import unittest
import pandas as pd
import numpy as np
from strategies.stochastic_oscillator_strategy import calculate_stochastic_oscillator, stochastic_oscillator_strategy
from config import STOCHASTIC_K_PERIOD, STOCHASTIC_D_PERIOD, STOCHASTIC_OVERBOUGHT , STOCHASTIC_OVERSOLD


class TestStochasticOscillatorStrategy(unittest.TestCase):
    def setUp(self):
        # Create sample data
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
        np.random.seed(42)  # Set seed for reproducibility
        prices = np.random.randn(len(dates)).cumsum() + 100
        self.sample_data = pd.DataFrame({
            'close': prices,
            'high': prices + np.random.rand(len(dates)),
            'low': prices - np.random.rand(len(dates))
        }, index=dates)

    def test_calculate_stochastic_oscillator(self):
        stoch = calculate_stochastic_oscillator(self.sample_data)
        
        self.assertIsInstance(stoch, pd.DataFrame)
        self.assertIn('%K', stoch.columns)
        self.assertIn('%D', stoch.columns)
        self.assertEqual(len(stoch), len(self.sample_data))
        self.assertTrue((stoch >= 0).all().all() and (stoch <= 100).all().all())

    def test_stochastic_oscillator_strategy(self):
        signals = stochastic_oscillator_strategy(self.sample_data)

        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('signal', signals.columns)
        self.assertIn('positions', signals.columns)
        self.assertIn('%K', signals.columns)
        self.assertIn('%D', signals.columns)

        # Check if signals are either -1, 0, or 1
        unique_signals = set(signals['signal'].unique())
        print("Unique Signals:", unique_signals)
        self.assertTrue(unique_signals.issubset({-1, 0, 1}))

        # Check if positions are -2, -1, 0, 1, or 2
        unique_positions = set(signals['positions'].unique())
        print("Unique Positions:", unique_positions)
        self.assertTrue(unique_positions.issubset({-2, -1, 0, 1, 2}))

if __name__ == '__main__':
    unittest.main()