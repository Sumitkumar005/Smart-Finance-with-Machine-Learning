# Using computational methods to fair-value options and other derivatives.

'''
Python script for option pricing using the Black-Scholes model. This model is used to estimate the fair value of European call and put options.
'''

# Import necessary libraries
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Function to calculate the Black-Scholes price for a European call option
def black_scholes_call(S, K, T, r, sigma):
    """
    S: Current stock price
    K: Strike price
    T: Time to expiration in years
    r: Risk-free interest rate
    sigma: Volatility of the underlying asset
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    return call_price

# Function to calculate the Black-Scholes price for a European put option
def black_scholes_put(S, K, T, r, sigma):
    """
    S: Current stock price
    K: Strike price
    T: Time to expiration in years
    r: Risk-free interest rate
    sigma: Volatility of the underlying asset
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
    return put_price

# Example parameters
S = 100  # Current stock price
K = 100  # Strike price
T = 1    # Time to expiration (1 year)
r = 0.05  # Risk-free interest rate (5%)
sigma = 0.2  # Volatility (20%)

# Calculate call and put option prices
call_price = black_scholes_call(S, K, T, r, sigma)
put_price = black_scholes_put(S, K, T, r, sigma)

# Print the results
print(f"European Call Option Price: {call_price:.2f}")
print(f"European Put Option Price: {put_price:.2f}")

# Plotting the option price sensitivity to volatility
volatilities = np.linspace(0.01, 1, 100)
call_prices = [black_scholes_call(S, K, T, r, vol) for vol in volatilities]
put_prices = [black_scholes_put(S, K, T, r, vol) for vol in volatilities]

plt.figure(figsize=(10, 6))
plt.plot(volatilities, call_prices, label='Call Option Price', color='blue')
plt.plot(volatilities, put_prices, label='Put Option Price', color='red')
plt.title('Option Price Sensitivity to Volatility')
plt.xlabel('Volatility')
plt.ylabel('Option Price')
plt.legend()
plt.grid(True)
plt.show()

'''
Explanation:

- Import Libraries: We import the necessary libraries, numpy for numerical calculations, scipy for statistical functions (needed for the cumulative distribution function), and matplotlib for plotting.
  
- Black-Scholes Functions: We define two functions for calculating the fair-value of European call and put options using the Black-Scholes model. The inputs include the current stock price, strike price, time to expiration, risk-free interest rate, and volatility of the underlying asset.

- Option Pricing Calculation: We set example values for the stock price, strike price, time to expiration, risk-free rate, and volatility. The call and put prices are then calculated and printed.

- Sensitivity Plot: To understand how the option price changes with volatility, we plot the option prices for different levels of volatility.

Note: In practice, this is a simple example for European options, and you might need to adjust it for more complex derivative pricing (e.g., American options, multi-asset options). More advanced models might use Monte Carlo simulations, binomial trees, or other methods depending on the complexity of the derivatives.
'''

