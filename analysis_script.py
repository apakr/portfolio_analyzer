import pandas as pd
import yfinance as yf
import numpy as np 
import platform
from datetime import datetime, timedelta
import pytz
import analysis_functs as funct

# Get operating system
os_name = platform.system()

# Get current date 
current_date = datetime.now(pytz.timezone('America/Chicago'))

# Analysis dates
end_date = (current_date-timedelta(days=1)).strftime("%Y-%m-%d") # yyyy-mm-dd
start_date = (current_date-timedelta(days=1)).replace(year=current_date.year-1).strftime("%Y-%m-%d") # accounts for leap years and also for yfinance issues with using day-of data
# currently set to 1 year befored

# start_date = "2023-11-16" # hard coded dates for analysis
# end_date = "2024-11-16"

# Load original portfolio
portf = pd.read_csv("pmt_portfolio.csv")

# Calculate metrics for original portfolio
original_metrics = funct.calculate_portfolio_metrics(portf, start_date, end_date)

# # Download S&P 500 (or another market index) data
market_data = yf.download('^GSPC', start=start_date, end=end_date, interval="1mo")['Adj Close']

# Calculate market returns at specified frequency (monthly right now)
market_returns = market_data.pct_change().dropna()

# Calculate expected return value for the market
market_ret = funct.calc_exp_ret(market_data)

# Fetch risk-free rate (10-Year Treasury Yield)
risk_free_data = yf.download('^TNX', start=start_date, end=end_date, interval="1mo")
risk_free_rate = risk_free_data['Adj Close'].dropna().iloc[-1].item() / 100

# Calculate Beta
original_beta = funct.calculate_beta(
    original_metrics[2], market_returns
    )

# Calculate Sharpe
original_sharpe = funct.calc_sharpe_ratio(
    original_metrics[0], risk_free_rate, original_metrics[1]
    )

# Calculate alpha for portfolio
original_alpha = float(funct.calculate_alpha(original_metrics[0],risk_free_rate,original_beta,market_ret).iloc[0])

# Display comparison
print("Original Portfolio:")
print(f"Expected Return: {original_metrics[0]:.4f}")
print(f"Portfolio Std Dev: {original_metrics[1]:.4f}")
print(f"Beta: {original_beta:.4f}")
print(f"Sharpe Ratio: {original_sharpe:.4f}")
print(f"Alpha: {original_alpha:.4f}")



