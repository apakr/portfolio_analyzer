import pandas as pd
import yfinance as yf
import numpy as np 
import platform
from datetime import datetime, timedelta
import pytz

# Get operating system
os_name = platform.system()

# Get current date 
current_date = datetime.now(pytz.timezone('America/New_York'))

# Analysis dates
end_date = (current_date-timedelta(days=1)).strftime("%Y-%m-%d") # yyyy-mm-dd
start_date = (current_date-timedelta(days=1)).replace(year=current_date.year-1).strftime("%Y-%m-%d") # accounts for leap years and also for yfinance issues with using day-of data
# currently set to 1 year befored

# start_date = "2023-11-16" # hard coded dates for analysis
# end_date = "2024-11-16"
 

# Load original portfolio
portf = pd.read_csv("pmt_portfolio.csv")

# Load additional portfolio // optional, can be used to compare proposed changes.
# additional_portf = pd.read_csv("fin456_portfolio_holdings_t1.csv")

def calculate_portfolio_metrics(portf):
    # Separate cash and stock rows
    cash_row = portf[portf['TICKER'] == 'USD']
    stock_rows = portf[portf['TICKER'] != 'USD'].copy()  # Ensure this is a copy

    # Fetch data for the tickers in the portfolio
    tickers = stock_rows['TICKER'].tolist()
    data = yf.download(tickers, start=start_date, end=end_date, interval="1mo")['Adj Close']

    # Calculate monthly returns
    returns = data.pct_change().dropna()

    # Calculate total investment in each stock
    current_prices = data.iloc[-1]  # Last row gives the latest prices
    stock_rows['Investment'] = stock_rows['QUANTITY'].values * current_prices.values

    # Check if cash row is empty and handle accordingly
    if not cash_row.empty:
        cash_quantity = cash_row['QUANTITY'].iloc[0]
    else:
        cash_quantity = 0

    # Add the cash row back for total investment
    total_investment = stock_rows['Investment'].sum() + cash_quantity

    # Calculate weights
    stock_rows['Weight'] = stock_rows['Investment'] / total_investment

    # Handle cash weight separately
    cash_weight = cash_quantity / total_investment if cash_quantity > 0 else 0

    # Calculate weighted monthly returns for the portfolio
    weighted_returns = (returns * stock_rows.set_index('TICKER')['Weight']).sum(axis=1)

    # Adjust weighted returns to include cash
    weighted_returns = weighted_returns * (1 - cash_weight)

    # Expected annual return
    # expected_return = weighted_returns.mean() * 12 # using arithmetric mean
    expected_return = (1+weighted_returns).prod()**(12/weighted_returns.size) - 1 # using geometric mean

    # Portfolio variance and standard deviation (risk)
    portfolio_variance = np.dot(stock_rows.set_index('TICKER')['Weight'].T, 
                                np.dot(returns.cov() * 12, 
                                       stock_rows.set_index('TICKER')['Weight']))
    portfolio_variance *= (1 - cash_weight)**2  # Adjust for cash weight
    portfolio_std_dev = np.sqrt(portfolio_variance)

    return expected_return, portfolio_std_dev, weighted_returns, cash_weight

# Function to calulate sharpe ratio
def calc_sharpe_ratio(exp_ret, risk_free_rate, portfolio_std_dev):
    sharpe_ratio = (exp_ret - risk_free_rate) / portfolio_std_dev
    return sharpe_ratio

# Function to calculate Beta and Sharpe Ratio
def calculate_beta(portfolio_returns, market_returns):
    aligned_portfolio_returns, aligned_market_returns = portfolio_returns.align(market_returns, join='inner')

    # Ensure aligned_market_returns is a Series
    if isinstance(aligned_market_returns, pd.DataFrame):
        aligned_market_returns = aligned_market_returns.squeeze()  # Convert single-column DataFrame to Series

    cov_matrix = np.cov(aligned_portfolio_returns, aligned_market_returns)  # returns cov matrix of [[var(apr), cov(apr,amr)],[cov(amr,apr),var(amr)]]

    # Calculate beta
    beta = cov_matrix[0, 1] / cov_matrix[1, 1] # divides cov(apr,amr) by var(amr)
    return beta


# Function to calculate expected return for a given ticker - "easy" because you just give it a name and it finds the data for you
def easy_exp_ret(ticker, start_date, end_date): # helper function, returns expected return for a given security. doesn't work for a list of securities.

    # Download S&P 500 (or another market index) data
    market_data = yf.download(ticker, start=start_date, end=end_date, interval="1mo")['Adj Close']

    # Calculate market monthly returns
    market_returns = market_data.pct_change().dropna()

    expected_return = (1+market_returns).prod()**(12/market_returns.size) - 1 # using geometric mean

    return expected_return

# Function to calculate expected return given some data - could be called "hard" because you need to already have the data instead of yfinance doing the work
def calc_exp_ret(data):

    returns = data.pct_change().dropna()

    expected_return = (1+returns).prod()**(12/returns.size) - 1 # using geometric mean, and also set to monthly right now

    return expected_return # returns a single number value for the expected return given data

# Function to calculate the Alpha
def calculate_alpha(portf_ret, risk_free_rate, beta, market_ret):
    # Alpha = R - Rf - beta (Rm - Rf) ... R is the portf_ret, Rf is the risk_free_rate, beta is the systematic risk of the portfolio, Rm is the market return.
    alpha = portf_ret - risk_free_rate - beta*(market_ret-risk_free_rate)
    return alpha

# Calculate metrics for original portfolio
original_metrics = calculate_portfolio_metrics(portf)

# # Download S&P 500 (or another market index) data
market_data = yf.download('^GSPC', start=start_date, end=end_date, interval="1mo")['Adj Close']

# Calculate market returns at specified frequency (monthly right now)
market_returns = market_data.pct_change().dropna()

# Calculate expected return value for the market
market_ret = calc_exp_ret(market_data)

# Fetch risk-free rate (10-Year Treasury Yield)
risk_free_data = yf.download('^TNX', start=start_date, end=end_date, interval="1mo")
risk_free_rate = risk_free_data['Adj Close'].dropna().iloc[-1].item() / 100

# Calculate Beta
original_beta = calculate_beta(
    original_metrics[2], market_returns
    )

# Calculate Sharpe
original_sharpe = calc_sharpe_ratio(
    original_metrics[0], risk_free_rate, original_metrics[1]
    )

# Calculate alpha for portfolio
original_alpha = float(calculate_alpha(original_metrics[0],risk_free_rate,original_beta,market_ret).iloc[0])

# Calculate metrics for additional portfolio # uncomment for additional portfolio # needs to be updated for function changes
# additional_metrics = calculate_portfolio_metrics(additional_portf)

# additional_beta, additional_sharpe = calculate_beta(  # uncomment for additional portfolio
#     additional_metrics[2], market_returns, risk_free_rate, additional_metrics[1], additional_metrics[0]
# )

# additional_alpha = calculate_alpha(additional_metrics[0],risk_free_rate,additional_beta,market_ret)

# Display comparison
print("Original Portfolio:")
print(f"Expected Return: {original_metrics[0]:.4f}")
print(f"Portfolio Std Dev: {original_metrics[1]:.4f}")
print(f"Beta: {original_beta:.4f}")
print(f"Sharpe Ratio: {original_sharpe:.4f}")
print(f"Alpha: {original_alpha:.4f}")

# print("\nAdditional Portfolio:") # uncomment for additional portfolio
# print(f"Expected Return: {additional_metrics[0]:.4f}")
# print(f"Portfolio Std Dev: {additional_metrics[1]:.4f}")
# print(f"Beta: {additional_beta:.4f}")
# print(f"Sharpe Ratio: {additional_sharpe:.4f}")
# print(f"Alpha: {additional_alpha:.4f}")


