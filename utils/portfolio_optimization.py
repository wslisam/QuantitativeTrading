import numpy as np
import pandas as pd
from scipy.optimize import minimize

def portfolio_performance(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return portfolio_return, portfolio_volatility

def neg_sharpe_ratio(weights, returns, risk_free_rate=0.02):
    p_ret, p_vol = portfolio_performance(weights, returns)
    return -(p_ret - risk_free_rate) / p_vol

def optimize_portfolio(returns, min_weight=0.05, max_weight=0.4, min_assets=3):
    num_assets = returns.shape[1]
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
        {'type': 'ineq', 'fun': lambda x: np.sum(x > min_weight) - min_assets}  # At least min_assets have weight > min_weight
    ]
    
    # Bounds
    bounds = tuple((min_weight, max_weight) for _ in range(num_assets))
    
    # Initial guess: equal weights
    init_guess = np.array([1.0 / num_assets] * num_assets)
    
    # Optimize
    result = minimize(neg_sharpe_ratio, init_guess, args=(returns,),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    
    # Ensure the result meets the minimum assets constraint
    weights = result.x
    if np.sum(weights > min_weight) < min_assets:
        # If constraint not met, force top min_assets to have at least min_weight
        top_indices = np.argsort(weights)[-min_assets:]
        weights[top_indices] = np.maximum(weights[top_indices], min_weight)
        weights = weights / np.sum(weights)  # Renormalize
    
    return weights

def safe_optimize_portfolio(returns, min_weight=0.05, max_weight=0.4, min_assets=3):
    try:
        if returns.empty or returns.isna().all().all():
            raise ValueError("Returns data is empty or contains only NaN values")
        
        if (returns == 0).all().all():
            raise ValueError("Returns data contains only zeros")

        weights = optimize_portfolio(returns, min_weight, max_weight, min_assets)
        return weights
    except Exception as e:
        print(f"Error in portfolio optimization: {str(e)}")
        return np.full(returns.shape[1], 1/returns.shape[1])  # Equal weights as fallback