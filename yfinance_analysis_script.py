import pandas as pd
import yfinance as yf
import numpy as np 

# Load original portfolio
portf = pd.read_csv("portfolio_holdings.csv")

# Load additional portfolio
additional_portf = pd.read_csv("")








cash_row = portf[portf['TICKER'] == 'USD']
portf = portf[portf['TICKER'] != 'USD']

# Fetch data for the tickers in the portfolio
tickers = portf['TICKER'].tolist()
data = yf.download(tickers, start="2019-01-01", end="2024-01-01", interval="1mo")['Adj Close']

# Calculate monthly returns
returns = data.pct_change().dropna()

# Calculate total investment in each stock
current_prices = data.iloc[-1]  # Last row gives the latest prices
portf['Investment'] = portf['QUANTITY'] * current_prices.values

# Add the cash row back for total investment
total_investment = portf['Investment'].sum() + cash_row['QUANTITY'].iloc[0]

# Calculate weights
portf['Weight'] = portf['Investment'] / total_investment

# Add a weight of cash (cash has zero returns)
cash_weight = cash_row['QUANTITY'].iloc[0] / total_investment

# Calculate weighted monthly returns for the portfolio
weighted_returns = (returns * portf.set_index('TICKER')['Weight']).sum(axis=1)

# Adjust weighted returns to include cash (cash return is zero)
weighted_returns = weighted_returns * (1 - cash_weight)

# Expected annual return
expected_return = weighted_returns.mean() * 12

# Portfolio variance and standard deviation (risk)
portfolio_variance = np.dot(portf.set_index('TICKER')['Weight'].T, 
                            np.dot(returns.cov() * 12, 
                                   portf.set_index('TICKER')['Weight']))
portfolio_variance *= (1 - cash_weight)**2  # Adjust for cash weight
portfolio_std_dev = np.sqrt(portfolio_variance)

# Download S&P 500 (or another market index) data
market_data = yf.download('^GSPC', start="2019-01-01", end="2024-01-01", interval="1mo")['Adj Close']

# Calculate market monthly returns
market_returns = market_data.pct_change().dropna()

# Align both series and drop NaNs
aligned_portfolio_returns, aligned_market_returns = weighted_returns.align(market_returns, join='inner')

# Ensure no missing values remain after alignment
aligned_portfolio_returns = aligned_portfolio_returns.dropna()
aligned_market_returns = aligned_market_returns.dropna()

# Ensure aligned_market_returns is a Series
if isinstance(aligned_market_returns, pd.DataFrame):
    aligned_market_returns = aligned_market_returns.squeeze()  # Convert single-column DataFrame to Series

cov_matrix = np.cov(aligned_portfolio_returns, aligned_market_returns)

# Calculate beta
beta = cov_matrix[0, 1] / cov_matrix[1, 1]

# We need risk free frate for calculating the sharpe ratio, so we're just gonna use yfinance and fetch TNX. If you want perfectly accurate sharpe calculation you can use U.S. treasury data from the internet for that. 10-yr bonds.
# Fetch 10-Year Treasury Yield (^TNX)
risk_free_data = yf.download('^TNX', start="2019-01-01", end="2024-01-01", interval="1mo")

# Ensure you get the last valid yield as a scalar
risk_free_rate = risk_free_data['Adj Close'].dropna().iloc[-1] / 100  # Ensure single value and convert

# Sharpe Ratio
sharpe_ratio = (expected_return - risk_free_rate)/portfolio_std_dev

print(expected_return, portfolio_std_dev, beta, sharpe_ratio)