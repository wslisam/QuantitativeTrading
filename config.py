# Trading parameters
INITIAL_CAPITAL = 100000.0

# Data parameters
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2023-04-30"

# Strategy parameters

# Moving Average Crossover Strategy
MOVING_AVERAGE_SHORT_WINDOW = 50
MOVING_AVERAGE_LONG_WINDOW = 200

# RSI Strategy
RSI_WINDOW = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Bollinger Bands Strategy
BOLLINGER_WINDOW = 20
BOLLINGER_NUM_STD = 2

# Machine Learning Strategy parameters
ML_LOOKBACK = 30
ML_TEST_SIZE = 0.2
ML_N_ESTIMATORS = 100

# Moving Average Convergence Divergence
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Stochastic Oscillator Strategy parameters
STOCHASTIC_K_PERIOD = 14
STOCHASTIC_D_PERIOD = 3
STOCHASTIC_OVERBOUGHT = 80
STOCHASTIC_OVERSOLD = 20