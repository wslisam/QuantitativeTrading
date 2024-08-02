# strategies/ml_strategy.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.indicators import calculate_rsi

def ml_strategy(data, lookback=30, test_size=0.2, n_estimators=100):
    if len(data) <= lookback:
        raise ValueError(f"Not enough data. Required: >{lookback}, Provided: {len(data)}")

    # Feature engineering
    data['Returns'] = data['Close'].pct_change()
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['RSI'] = calculate_rsi(data['Close'], window=14)
    
    # Drop NaN values
    data.dropna(inplace=True)
    
    # Create features and target
    X = data[['Returns', 'MA5', 'MA20', 'RSI']].values
    y = (data['Close'].shift(-1) > data['Close']).astype(int).values
    
    # Ensure X and y have the same length
    X = X[:-1]
    y = y[:-1]
    
    if len(X) != len(y):
        raise ValueError(f"Mismatch in lengths. X: {len(X)}, y: {len(y)}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    predictions = model.predict(X_test_scaled)
    
    # Create a DataFrame with the results
    results = data.iloc[-len(X_test):].copy()
    results['Signal'] = predictions
    results['Position'] = results['Signal'].shift(1)
    results['Returns'] = np.log(results['Close'] / results['Close'].shift(1))
    results['Strategy_Returns'] = results['Position'] * results['Returns']
    
    return results