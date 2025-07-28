# -*- coding: utf-8 -*-
"""
Created on Tue Nov 31 01:12:27 2024

@author: Yash Singhal
"""

import numpy as np
import matplotlib.pyplot as plt



def simulate_gbm(S0,mu,sigma,T,n_steps,n_sims):
    """
    Simulates Geometric Brownian Motion paths
    Parameters
    ----------
    S0 : Float
        Initial Stock Price
    mu : Float
        Drift Coefficient (risk-free rate)
    sigma : Float
        Diffusion Coefficient(or volatility)
    T : float
        Time period
    n_steps : int
        Number of time steps in each path
    n_sims : int
        Number of simulation paths to generate

    Returns
    -------
    numpy.ndarray: Array of simulated asset paths. Shape (n_steps + 1, n_sims).
    """
    
    print('Simulating GBM paths')
    dt = T/n_steps
    
    #Generate Random Movements (Random Walk based movement)
    random_walk = np.random.standard_normal((n_steps,n_sims))
    
    
    paths = np.zeros((n_steps+1,n_sims))
    paths[0] = S0
    
    
    #Euler-Maruyama Discretization of log price SDE
    for t in range(1,n_steps+1):
        paths[t] = paths[t-1]*np.exp(
                (mu-0.5*sigma*sigma)*dt + sigma * np.sqrt(dt)*random_walk[t-1]
            )
        
    return paths


if __name__ == '__main__':
    #______________________________Configuration_______________________________
    S0 = 100.0
    mu = 0.05
    sigma = 0.20
    T = 1.0
    n_steps = 252
    n_sims = 1000
    
    sim_paths = simulate_gbm(S0, mu, sigma, T, n_steps, n_sims)
    
    
    # --- Plot the results ---
    print("Plotting results...")
    plt.figure(figsize=(10, 6))
    
    # Plot a subset of the paths to keep the plot clean
    plt.plot(sim_paths[:, :10])
    
    plt.title(f'GBM Price Path Simulation (10 of {n_sims} paths)')
    plt.xlabel('Time Steps (Days)')
    plt.ylabel('Asset Price')
    plt.grid(True)
    plt.savefig(r'Simulated Paths.png')
    plt.show()

    print("\nGBM Simulation complete. This script provides the foundation for our option pricing engine.")
    