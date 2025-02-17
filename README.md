# Calculation Functionality 
analysis_script is a script that you can use to figure out important porfolio management statistics for a given set of equities:
1. Expected Return
2. Standard Deviation
3. Beta
4. Sharpe Ratio
5. Alpha

You can choose what time frame you use for calculation as well as what frequency of returns:
```
# Analysis dates
end_date = (current_date-timedelta(days=1)).strftime("%Y-%m-%d") # yyyy-mm-dd
start_date = (current_date-timedelta(days=1)).replace(year=current_date.year-1).strftime("%Y-%m-%d") 
```
Adjusting the "-1" in the 3rd line adjusts how many years back you run the analysis. Found on line 15 in analysis_script.py.

```
# Frequency
freq = '1mo'
```
Found on line 20 in analysis_scripts.py.


You need a CSV with the stocks/funds you own and the number of each security that you own - look at fin456_portfolio_holdings_t0 for an example. 
The program uses yfinance to fetch the data.

The scripts analysis_script.py and analysis_functs.py do the work for this functionality, with the former being the actual script that is run and the latter holding helper functions that do a lot of the work "under the hood" for the primary script.

<br>

# Optimization Functionality

It is often a wise decision to know the optimal sharpe ratio for a portfolio consisting of a select number of equities. Knowing what your current return to risk profile looks like compared to what constitution of securities would optimize it helps inform your discretionary decisionmaking as a manager of a portfolio. 

You can compare portfolio movements you think are good vs. what the statistical projection of returns has to say. 

## Monte Carlo Optimization

