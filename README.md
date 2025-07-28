# High-Performance Exotic Option Pricing Engine

This repository contains a Python-based engine for pricing path-dependent exotic options using a high-performance Monte Carlo simulation. The key feature of this project is the implementation of a machine learning-enhanced variance reduction technique (Control Variates via Linear Regression) to achieve faster and more accurate price convergence.

The project demonstrates a deep understanding of core quantitative finance concepts, including stochastic calculus, risk-neutral pricing, and advanced computational methods.

<img width="1248" height="716" alt="image" src="https://github.com/user-attachments/assets/6bb1cc79-460b-4cec-9326-99be90ad9988" />
~Source:Stochastic Calculus for Finance II
        Continuous-Time Models
        Steven E. Shreve

## Key Features

- **Stochastic Process Simulation:** Implements a Geometric Brownian Motion (GBM) engine to simulate risk-neutral price paths of the underlying asset.
- **Exotic Option Pricing:** Prices Asian options (both calls and puts), whose payoffs depend on the average asset price, making them unsolvable with closed-form formulas.
- **ML-Enhanced Variance Reduction:** Uses Linear Regression to find the optimal hedging coefficient for the Control Variates technique. This significantly reduces the variance of the Monte Carlo estimator, a critical aspect of computational efficiency.
- **Performance Analysis:** Directly compares the variance of the standard Monte Carlo method against the ML-enhanced method to quantify the efficiency gains.

## Core Concepts Demonstrated

This project is a practical application of several fundamental quantitative finance concepts:

- **Stochastic Calculus:** The implementation of the Geometric Brownian Motion SDE.
- **Risk-Neutral Pricing:** The core principle that derivatives are priced as the discounted expectation of future payoffs under a risk-neutral measure.
- **Monte Carlo Methods:** A powerful numerical technique for estimating expectations when closed-form solutions are not available.
- **Variance Reduction Techniques:** Advanced methods like Control Variates used to improve the efficiency and accuracy of Monte Carlo simulations.
- **The "Greeks":** While not fully implemented in the final script, the foundation is laid for calculating risk sensitivities like Delta via numerical methods.

## Example Results

<img width="720" height="432" alt="Simulated Paths" src="https://github.com/user-attachments/assets/482e794e-3b55-48e8-89b1-e5893f6b429f" />


Running the engine demonstrates the power of the ML-enhanced approach. The output clearly shows that the variance of the control variate estimator is significantly lower than that of the standard Monte Carlo estimator, often achieving a **variance reduction of over 65%**. This means a far more stable and accurate price can be achieved with the higher number of simulations.

```
--- Results Comparison ---
Standard Monte Carlo Price: $5.7666 (Variance: 70.6235)
ML-Enhanced CV Price:     $5.5433 (Variance: 20.8284)
Variance Reduction Achieved: 70.51%
```

## Future Work & Potential Enhancements

This engine provides a strong foundation for several advanced extensions:

- **Implement a Stochastic Volatility Model:** Replace the GBM simulator with a more realistic model like the **Heston Model**, which allows volatility to be a random process itself.
- **Efficient Greek Calculation:** Implement "Pathwise Derivatives" to calculate Delta and other Greeks within a single simulation run, avoiding the computationally expensive "bump and revalue" method.
- **Price Other Exotic Options:** Extend the engine's capabilities to price other path-dependent options like Barrier or Lookback options.

---
*This project was developed as an exercise in quantitative financial engineering.*
