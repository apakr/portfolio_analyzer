import pandas as pd # version 2.2.3
import yfinance as yf # version 0.2.49
import numpy as np # version 2.1.3
import platform # platformdirs version 4.3.6
from datetime import datetime, timedelta 
import pytz # version 2024.2
import analysis_functs as funct
import matplotlib.pyplot as plt # version 3.10.0

# Get operating system
os_name = platform.system()

# Get current date 
current_date = datetime.now(pytz.timezone('America/Chicago'))

# Analysis dates
end_date = (current_date-timedelta(days=1)).strftime("%Y-%m-%d") # yyyy-mm-dd
start_date = (current_date-timedelta(days=1)).replace(year=current_date.year-1).strftime("%Y-%m-%d") # accounts for leap years and also for yfinance issues with using day-of data
# currently set to 1 year befored

# Frequency
freq = '1d'
# 5m for 5 minutes, 1d for day, 1wk for week, 1mo for month, etc.

# start_date = "2023-11-16" # hard coded dates for analysis
# end_date = "2024-11-16"

# Load original portfolio
portf = pd.read_csv("portfolios/pmt_portfolio.csv")

# Calculate metrics for original portfolio
original_metrics = funct.calculate_portfolio_metrics(portf, start_date, end_date)

# # Download S&P 500 (or another market index) data
market_data = yf.download('^GSPC', start=start_date, end=end_date, interval=freq)['Close']

# Calculate market returns at specified frequency (monthly right now)
market_returns = market_data.pct_change().dropna()

# Calculate expected return value for the market
market_ret = funct.calc_exp_ret(market_data)

# Fetch risk-free rate (10-Year Treasury Yield)
risk_free_data = yf.download('^TNX', start=start_date, end=end_date, interval=freq)
risk_free_rate = risk_free_data['Close'].dropna().iloc[-1].item() / 100

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

# print(original_metrics[4])

### Optional plots
plot = True
if (plot):

    fig, axes = plt.subplots(3,1,figsize=(10,15))

    # Individual pct change in stocks since start date
    # plt.figure()
    indv_stocks_growth = original_metrics[4].div(original_metrics[4].iloc[0], axis = 1)
    indv_stocks_growth.plot(ax=axes[0])
    axes[0].set_title("Individual pct change since start date")
    # indv_stocks_growth.plot(figsize=(10,8))
    # plt.title("Indivdual pct change since start date")
    # plt.show()

    # Individual total investment changes in each stock
    # plt.figure()
    indv_stocks_growth_price = original_metrics[4].mul(original_metrics[5].set_index("TICKER")["QUANTITY"], axis=1)
    indv_stocks_growth_price.plot(ax=axes[1])
    axes[1].set_title("Individual total investment changes since start date")

    # isgp = indv_stocks_growth_price.plot(figsize=(10,8))
    # plt.title("Indivudal total change since start date")
    # plt.show()

    # Total portfolio pct change since start date
    indv_stocks_growth_price.sum(axis=1).plot(ax=axes[2])
    axes[2].set_title("Portfolio over time")
    axes[2].set_ylim(0,None)

    plt.tight_layout()
    plt.show()
