# -*- coding: utf-8 -*-
"""
Created on Tue Dec4 12:05:50 2024

@author: Yash Singhal
"""

import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from Simulate_GBM import simulate_gbm

def bs_european_price(s0, K, r, sigma, T, option_type='call'):
    """
    Calculates price of European option using Black Scholes Method
    Parameters
    ----------
    S0 : Float
        Initial Stock Price
    r : Float
        Drift Coefficient (risk-free rate)
    sigma : Float
        Diffusion Coefficient(or volatility)
    T : float
        Time period
    K : Float
        Strike Price
    option_type : string
        Call or Put The default is 'call'.

    Returns
    -------
    price : float
        Black Scholes based option price

    """
    
    
    d1 = (np.log(s0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = (s0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type == 'put':
        price = (K * np.exp(-r * T) * norm.cdf(-d2) - s0 * norm.cdf(-d1))
    return price


def price_asian_option_cv(S0, K, r, sigma, T, n_steps, n_sims, option_type = 'call',n_regression_paths=10000):
    """
    Prices an Asian option using Monte Carlo with ML-enhanced control variates.
    
    Parameters
    ----------
    S0 : Float
        Initial Stock Price
    r : Float
        Drift Coefficient (risk-free rate)
    sigma : Float
        Diffusion Coefficient(or volatility)
    T : float
        Time period
    n_steps : int
        Number of time steps in each path
    n_sims : int
        Number of simulation paths to generate to actually calculate price
    K : Float
        Strike Price
    option_type : string
        Call or Put The default is 'call'.
    n_regression_paths : int
        Number of regression paths to train model on. The default is 10000.

    Returns
    -------
    Final option price

    """
    
    print(f"\n--- ML-Enhanced Control Variate Pricing for an Asian {option_type.upper()} ---")
    
    reg_paths = simulate_gbm(S0, r, sigma, T, n_steps, n_regression_paths)
    

    if option_type == 'call':
        asian_payoffs_reg = np.maximum(np.mean(reg_paths[1:], axis=0) - K, 0)
        european_payoffs_reg = np.maximum(reg_paths[-1] - K, 0)
    elif option_type == 'put':
        asian_payoffs_reg = np.maximum(K - np.mean(reg_paths[1:], axis=0), 0)
        european_payoffs_reg = np.maximum(K - reg_paths[-1], 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
        
        
    model = LinearRegression()
    model.fit(european_payoffs_reg.reshape(-1,1),asian_payoffs_reg)
    b_optimal = model.coef_[0]
    print(f"Optimal control variate coefficient 'b' found via ML: {b_optimal:.4f}")
    
    print(f"\nStep 2: Running {n_sims} main simulations for final pricing...")
    main_paths = simulate_gbm(S0, r, sigma, T, n_steps, n_sims)
    
    if option_type == 'call':
        asian_payoffs_main = np.maximum(np.mean(main_paths[1:], axis=0) - K, 0)
        european_payoffs_main = np.maximum(main_paths[-1] - K, 0)
    else: # 'put'
        asian_payoffs_main = np.maximum(K - np.mean(main_paths[1:], axis=0), 0)
        european_payoffs_main = np.maximum(K - main_paths[-1], 0)
        
    expected_european_price = bs_european_price(S0, K, r, sigma, T, option_type)
    
    control_variate_payoffs = asian_payoffs_main - b_optimal * (european_payoffs_main - expected_european_price)
    
    standard_mc_price = np.mean(asian_payoffs_main) * np.exp(-r * T)
    standard_variance = np.var(asian_payoffs_main)
    
    cv_mc_price = np.mean(control_variate_payoffs) * np.exp(-r * T)
    cv_variance = np.var(control_variate_payoffs)
    
    variance_reduction = (1 - cv_variance / standard_variance) * 100
    
    print("\n--- Results Comparison ---")
    print(f"Standard Monte Carlo Price: ${standard_mc_price:.4f} (Variance: {standard_variance:.4f})")
    print(f"ML-Enhanced CV Price:     ${cv_mc_price:.4f} (Variance: {cv_variance:.4f})")
    print(f"Variance Reduction Achieved: {variance_reduction:.2f}%")
    
    
if __name__ == '__main__':
    # --- Configuration ---
    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.20
    T = 1.0
    n_steps = 252
    n_sims = 500000
    
    # --- Run for a CALL option ---
    call_price = price_asian_option_cv(S0, K, r, sigma, T, n_steps, n_sims, option_type='call')
    
    # --- Run for a PUT option ---
    put_price = price_asian_option_cv(S0, K, r, sigma, T, n_steps, n_sims, option_type='put')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    