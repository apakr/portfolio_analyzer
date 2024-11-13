import pandas as pd
import yfinance as yf
import numpy as np 

# Load original portfolio
portf = pd.read_csv("fin456_portfolio_holdings_t1.csv")

# Load additional portfolio
additional_portf = pd.read_csv("cg_sector_holdings_t1.csv")

def calculate_portfolio_metrics(portf):
    # Separate cash and stock rows
    cash_row = portf[portf['TICKER'] == 'USD']
    stock_rows = portf[portf['TICKER'] != 'USD'].copy()  # Ensure this is a copy

    # Fetch data for the tickers in the portfolio
    tickers = stock_rows['TICKER'].tolist()
    data = yf.download(tickers, start="2023-11-11", end="2024-11-11", interval="1mo")['Adj Close']

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
    expected_return = weighted_returns.mean() * 12

    # Portfolio variance and standard deviation (risk)
    portfolio_variance = np.dot(stock_rows.set_index('TICKER')['Weight'].T, 
                                np.dot(returns.cov() * 12, 
                                       stock_rows.set_index('TICKER')['Weight']))
    portfolio_variance *= (1 - cash_weight)**2  # Adjust for cash weight
    portfolio_std_dev = np.sqrt(portfolio_variance)

    return expected_return, portfolio_std_dev, weighted_returns, cash_weight

# Calculate metrics for original portfolio
original_metrics = calculate_portfolio_metrics(portf)

# Calculate metrics for additional portfolio
additional_metrics = calculate_portfolio_metrics(additional_portf)

# # Download S&P 500 (or another market index) data
market_data = yf.download('^GSPC', start="2023-11-11", end="2024-11-11", interval="1mo")['Adj Close']

# # Calculate market monthly returns
market_returns = market_data.pct_change().dropna()

# Fetch risk-free rate (10-Year Treasury Yield)
risk_free_data = yf.download('^TNX', start="2023-11-11", end="2024-11-11", interval="1mo")
risk_free_rate = risk_free_data['Adj Close'].dropna().iloc[-1].item() / 100

# Function to calculate Beta and Sharpe Ratio
def calculate_beta_sharpe(portfolio_returns, market_returns, risk_free_rate, portfolio_std_dev, expected_return):
    aligned_portfolio_returns, aligned_market_returns = portfolio_returns.align(market_returns, join='inner')

    # Ensure aligned_market_returns is a Series
    if isinstance(aligned_market_returns, pd.DataFrame):
        aligned_market_returns = aligned_market_returns.squeeze()  # Convert single-column DataFrame to Series

    cov_matrix = np.cov(aligned_portfolio_returns, aligned_market_returns)

    # Calculate beta
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]

    # Sharpe Ratio
    sharpe_ratio = (expected_return - risk_free_rate) / portfolio_std_dev
    return beta, sharpe_ratio


# Calculate Beta and Sharpe Ratio for both portfolios
original_beta, original_sharpe = calculate_beta_sharpe(
    original_metrics[2], market_returns, risk_free_rate, original_metrics[1], original_metrics[0]
)
additional_beta, additional_sharpe = calculate_beta_sharpe(
    additional_metrics[2], market_returns, risk_free_rate, additional_metrics[1], additional_metrics[0]
)

# Display comparison
print("Original Portfolio:")
print(f"Expected Return: {original_metrics[0]:.4f}")
print(f"Portfolio Std Dev: {original_metrics[1]:.4f}")
print(f"Beta: {original_beta:.4f}")
print(f"Sharpe Ratio: {original_sharpe:.4f}")

print("\nAdditional Portfolio:")
print(f"Expected Return: {additional_metrics[0]:.4f}")
print(f"Portfolio Std Dev: {additional_metrics[1]:.4f}")
print(f"Beta: {additional_beta:.4f}")
print(f"Sharpe Ratio: {additional_sharpe:.4f}")

