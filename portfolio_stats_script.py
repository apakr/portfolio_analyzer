import pandas as pd

# Load the uploaded CSV file to inspect its structure
uploaded_file_path = 'adm_monthly_historical-data-11-12-2024.csv'
adm_data = pd.read_csv(uploaded_file_path)

print(adm_data.head())

# Calculate expected return and standard deviation
adm_data['Monthly Return'] = adm_data['%Chg'].str.replace('%', '').astype(float) / 100
expected_return = adm_data['Monthly Return'].mean()
std_dev = adm_data['Monthly Return'].std()

print(f"Expected Monthly Return: {expected_return}")
print(f"Standard Deviation of Monthly Return: {std_dev}")
