import pandas as pd
import analysis_functs as funct
import numpy as np
import yfinance as yf

start_date = "2023-11-16" # hard coded dates for analysis
end_date = "2024-11-16"

portf = pd.read_csv("simple_portf.csv")

