# -*- coding: utf-8 -*-
"""
Created on Tue Dec 1 13:26:18 2024

@author: Yash Singhal
"""

from Simulate_GBM import simulate_gbm
import numpy as np
from tqdm import tqdm

def price_asian_option(S0, K, r, sigma, T, n_steps, n_sims, option_type='call'):
    """
    Prices a standard Asian option using Monte Carlo Simulation
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
        Number of simulation paths to generate
    K : Float
        Strike Price
    option_type : string
        Call or Put The default is 'call'.
    Returns
    -------
    Option Price
    """
    
    print(f'Pricing Asian {option_type} option with standard Monte Carlo...')
    
    paths = simulate_gbm(S0, r, sigma, T, n_steps, n_sims)
    
    
    #Calculate Average price for each simulated path
    #EXcluding the initial Value:
    avg_price = np.mean(paths[1:],axis = 0)
    
    #Calculate option Pay-Off
    if option_type == 'call':
        payoffs = np.maximum(avg_price - K,0)
    elif option_type == 'put':
        payoffs = np.maximum(K - avg_price,0)
    else:
        print('Error Option Must either be call or put')
        
    #Discount the payoff to present Day
    option_price = np.mean(payoffs) * np.exp(-r*T)
    
    return option_price




if __name__ == '__main__':
    #______________________________Configuration_______________________________
    S0 = 100.0
    r = 0.05
    sigma = 0.20
    T = 1.0
    n_steps = 252
    n_sims = 1000
    K = 100.0 # At the money 
    
    
    # Price the option
    price = price_asian_option(S0, K, r, sigma, T, n_steps, n_sims)
    print(f'\nStandard Monte Carlo Price of the Asian Option: ${price:.4f}')
    
    # --- Calculate Delta Numerically (Bump-and-Revalue) ---
    print("\nCalculating Delta...")
    ds = 0.1  # A small change in the stock price
    
    # Price the option with a slightly higher initial price
    price_up = price_asian_option(S0, K, r, sigma, T, n_steps, n_sims)
    
    # Price the option with a slightly lower initial price
    price_down = price_asian_option(S0, K, r, sigma, T, n_steps, n_sims)
    
    # The Delta is the change in price divided by the change in stock price
    delta = (price_up - price_down) / (2 * ds)
    print(f"Numerically Calculated Delta: {delta:.4f}")